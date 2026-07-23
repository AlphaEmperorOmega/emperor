from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class _BankDistributionSummary:
    per_slot_utilization: Tensor
    mean_per_sample_entropy: Tensor


@dataclass(frozen=True)
class _BankUtilizationMetrics:
    marginal_entropy: Tensor
    mean_per_sample_entropy: Tensor
    coefficient_of_variation: Tensor
    active_slots: Tensor
    dead_slot_fraction: Tensor
    maximum_utilization: Tensor
    minimum_utilization: Tensor
    per_slot_utilization: Tensor


@dataclass(frozen=True)
class _WeightBankTrackingContext:
    pl_module: LightningModule
    module_name: str
    metrics: _BankUtilizationMetrics
    experiment: object | None
    global_step: int


class _WeightBankDiagnostics:
    @classmethod
    def summarize(
        cls,
        bank_module: Module,
        bank_logits: Tensor,
    ) -> _BankDistributionSummary | None:
        from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
            WeightedBankDynamicBias,
        )

        from .._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from .._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        if isinstance(bank_module, SoftWeightedBankDynamicWeight):
            return cls.summarize_soft_weighted_bank(bank_module, bank_logits)
        if isinstance(bank_module, LayeredWeightedBankDynamicWeight):
            return cls.summarize_layered_weighted_bank(bank_module, bank_logits)
        if isinstance(bank_module, WeightedBankDynamicBias):
            return cls.summarize_weighted_bank_bias(bank_module, bank_logits)
        return None

    @staticmethod
    def distribution_entropy(distribution: Tensor, dimension: int) -> Tensor:
        safe_distribution = distribution.clamp_min(1e-9)
        return -(safe_distribution.log() * distribution).sum(dim=dimension)

    @classmethod
    def summarize_soft_weighted_bank(
        cls,
        bank_module: Module,
        bank_logits: Tensor,
    ) -> _BankDistributionSummary:
        reshaped_logits = bank_logits.view(
            -1,
            bank_module.depth_value,
            bank_module.input_dim,
            bank_module.expanded_bank_row_count,
        )
        bank_distribution = torch.softmax(reshaped_logits, dim=-1)
        return _BankDistributionSummary(
            per_slot_utilization=bank_distribution.mean(dim=(0, 1, 2)),
            mean_per_sample_entropy=cls.distribution_entropy(
                bank_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def summarize_layered_weighted_bank(
        cls,
        bank_module: Module,
        bank_logits: Tensor,
    ) -> _BankDistributionSummary:
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        reshaped_distribution = bank_distribution.view(
            -1,
            bank_module.depth_value,
            bank_module.input_dim,
            bank_module.bank_expansion_factor,
        )
        return _BankDistributionSummary(
            per_slot_utilization=reshaped_distribution.sum(dim=2).mean(dim=(0, 1)),
            mean_per_sample_entropy=cls.distribution_entropy(
                bank_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def summarize_weighted_bank_bias(
        cls,
        bank_module: Module,
        bank_logits: Tensor,
    ) -> _BankDistributionSummary:
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        flat_distribution = bank_distribution.reshape(
            -1,
            bank_module.bank_expansion_factor,
        )
        return _BankDistributionSummary(
            per_slot_utilization=flat_distribution.mean(dim=0),
            mean_per_sample_entropy=cls.distribution_entropy(
                flat_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def calculate_utilization(
        cls,
        distribution_summary: _BankDistributionSummary,
        dead_slot_utilization_floor: float,
    ) -> _BankUtilizationMetrics:
        utilization = distribution_summary.per_slot_utilization.float()
        coefficient_of_variation = utilization.new_zeros(())
        if utilization.numel() > 1:
            coefficient_of_variation = utilization.std() / utilization.mean().clamp_min(
                1e-6
            )
        return _BankUtilizationMetrics(
            marginal_entropy=cls.distribution_entropy(utilization, dimension=-1),
            mean_per_sample_entropy=(distribution_summary.mean_per_sample_entropy),
            coefficient_of_variation=coefficient_of_variation,
            active_slots=(utilization > dead_slot_utilization_floor).sum().float(),
            dead_slot_fraction=(
                (utilization <= dead_slot_utilization_floor).float().mean()
            ),
            maximum_utilization=utilization.max(),
            minimum_utilization=utilization.min(),
            per_slot_utilization=utilization,
        )


class WeightBankUtilizationMonitorCallback(Callback):
    """Log slot utilization for adaptive weighted-bank parameters."""

    DEAD_SLOT_UTILIZATION_FLOOR = 1e-4

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_slot_scalars: bool = False,
    ) -> None:
        super().__init__()
        self.__validate_positive("log_every_n_steps", log_every_n_steps)
        self.__validate_positive("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_slot_scalars = log_per_slot_scalars
        self._hooks: list[RemovableHandle] = []
        self._bank_modules: list[tuple[str, Module]] = []
        self._utilization_history: dict[str, MonitorTensorHistory] = {}
        self._last_bank_logits: dict[str, Tensor] = {}
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive(option_name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{option_name} must be greater than 0.")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()
        for module_name, bank_module in pl_module.named_modules():
            if not self.__is_weighted_bank_module(bank_module):
                continue
            self._bank_modules.append((module_name, bank_module))
            self._utilization_history[module_name] = MonitorTensorHistory(
                self.history_size
            )
            generator_model = bank_module.model
            self._hooks.append(
                generator_model.register_forward_hook(
                    self.__make_bank_logits_capture_hook(module_name)
                )
            )

    @staticmethod
    def __is_weighted_bank_module(module: Module) -> bool:
        from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
            WeightedBankDynamicBias,
        )

        from .._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from .._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        return isinstance(
            module,
            (
                LayeredWeightedBankDynamicWeight,
                SoftWeightedBankDynamicWeight,
                WeightedBankDynamicBias,
            ),
        )

    def __make_bank_logits_capture_hook(
        self,
        module_name: str,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def capture_bank_logits(
            _generator: Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            bank_logits = self.__extract_bank_logits(output)
            if bank_logits is not None:
                self._last_bank_logits[module_name] = bank_logits.detach()

        return capture_bank_logits

    @staticmethod
    def __extract_bank_logits(output: object) -> Tensor | None:
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        return hidden if torch.is_tensor(hidden) else None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            self._last_bank_logits.clear()
            return
        for module_name, bank_module in self._bank_modules:
            bank_logits = self._last_bank_logits.pop(module_name, None)
            if bank_logits is None:
                continue
            distribution_summary = _WeightBankDiagnostics.summarize(
                bank_module,
                bank_logits,
            )
            if distribution_summary is None:
                continue
            metrics = _WeightBankDiagnostics.calculate_utilization(
                distribution_summary,
                self.DEAD_SLOT_UTILIZATION_FLOOR,
            )
            context = self.__build_tracking_context(
                pl_module,
                module_name,
                metrics,
            )
            self.__track_weight_bank_utilization(context)

    @staticmethod
    def __build_tracking_context(
        pl_module: LightningModule,
        module_name: str,
        metrics: _BankUtilizationMetrics,
    ) -> _WeightBankTrackingContext:
        return _WeightBankTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            metrics=metrics,
            experiment=getattr(
                getattr(pl_module, "logger", None),
                "experiment",
                None,
            ),
            global_step=getattr(pl_module, "global_step", 0),
        )

    def __track_weight_bank_utilization(
        self,
        context: _WeightBankTrackingContext,
    ) -> None:
        self.__track_marginal_selection_entropy(context)
        self.__track_mean_per_sample_selection_entropy(context)
        self.__track_utilization_coefficient_of_variation(context)
        self.__track_active_slots(context)
        self.__track_dead_slot_fraction(context)
        self.__track_maximum_utilization(context)
        self.__track_minimum_utilization(context)
        self.__track_per_slot_utilization(context)
        self.__track_utilization_history(context)
        self.__track_utilization_histogram(context)
        self.__track_utilization_heatmap(context)

    @staticmethod
    def __track_marginal_selection_entropy(
        context: _WeightBankTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/selection_entropy_marginal",
            context.metrics.marginal_entropy,
        )

    @staticmethod
    def __track_mean_per_sample_selection_entropy(
        context: _WeightBankTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/selection_entropy_mean",
            context.metrics.mean_per_sample_entropy,
        )

    @staticmethod
    def __track_utilization_coefficient_of_variation(
        context: _WeightBankTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/utilization_coefficient_of_variation",
            context.metrics.coefficient_of_variation,
        )

    @staticmethod
    def __track_active_slots(context: _WeightBankTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/active_slots",
            context.metrics.active_slots,
        )

    @staticmethod
    def __track_dead_slot_fraction(context: _WeightBankTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/dead_slot_fraction",
            context.metrics.dead_slot_fraction,
        )

    @staticmethod
    def __track_maximum_utilization(context: _WeightBankTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/max_utilization",
            context.metrics.maximum_utilization,
        )

    @staticmethod
    def __track_minimum_utilization(context: _WeightBankTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/bank/min_utilization",
            context.metrics.minimum_utilization,
        )

    def __track_per_slot_utilization(
        self,
        context: _WeightBankTrackingContext,
    ) -> None:
        if not self.log_per_slot_scalars:
            return
        for slot_index, utilization in enumerate(context.metrics.per_slot_utilization):
            context.pl_module.log(
                f"{context.module_name}/bank/slot_{slot_index}/utilization",
                utilization,
            )

    def __track_utilization_history(
        self,
        context: _WeightBankTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        self._utilization_history[context.module_name].append(
            context.metrics.per_slot_utilization
        )

    def __track_utilization_histogram(
        self,
        context: _WeightBankTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/bank/histogram/utilization",
            context.metrics.per_slot_utilization,
            context.global_step,
        )

    def __track_utilization_heatmap(
        self,
        context: _WeightBankTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/bank/heatmap/utilization",
            self._utilization_history[context.module_name],
            context.global_step,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self.__cleanup()

    def __cleanup(self) -> None:
        for hook_handle in self._hooks:
            hook_handle.remove()
        self._hooks.clear()
        self._bank_modules.clear()
        self._utilization_history.clear()
        self._last_bank_logits.clear()
        self._emission_policy.clear()
