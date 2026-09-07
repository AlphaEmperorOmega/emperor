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
class _WeightBankMetric:
    suffix: str
    value: Tensor


@dataclass(frozen=True)
class _WeightBankDiagnosticFacts:
    scalars: tuple[_WeightBankMetric, ...]
    per_slot_utilization: Tensor


class _WeightBankDiagnostics:
    """Collect ordered weight-bank facts without delivery concerns."""

    __slots__ = ()

    def collect(
        self,
        bank_module: Module,
        bank_logits: Tensor,
        *,
        dead_slot_utilization_floor: float,
        include_per_slot_scalars: bool,
    ) -> _WeightBankDiagnosticFacts | None:
        distribution_summary = self.__summarize(bank_module, bank_logits)
        if distribution_summary is None:
            return None
        metrics = self.__calculate_utilization(
            distribution_summary,
            dead_slot_utilization_floor,
        )
        scalar_facts = [
            _WeightBankMetric(
                "selection_entropy_marginal",
                metrics.marginal_entropy,
            ),
            _WeightBankMetric(
                "selection_entropy_mean",
                metrics.mean_per_sample_entropy,
            ),
            _WeightBankMetric(
                "utilization_coefficient_of_variation",
                metrics.coefficient_of_variation,
            ),
            _WeightBankMetric("active_slots", metrics.active_slots),
            _WeightBankMetric("dead_slot_fraction", metrics.dead_slot_fraction),
            _WeightBankMetric("max_utilization", metrics.maximum_utilization),
            _WeightBankMetric("min_utilization", metrics.minimum_utilization),
        ]
        if include_per_slot_scalars:
            scalar_facts.extend(
                _WeightBankMetric(f"slot_{slot_index}/utilization", utilization)
                for slot_index, utilization in enumerate(metrics.per_slot_utilization)
            )
        return _WeightBankDiagnosticFacts(
            scalars=tuple(scalar_facts),
            per_slot_utilization=metrics.per_slot_utilization,
        )

    def __summarize(
        self,
        bank_module: Module,
        bank_logits: Tensor,
    ) -> _BankDistributionSummary | None:
        from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
            WeightedBankDynamicBias,
        )

        from .._biases.variants.matrix_mixture import MatrixBiasMixture
        from .._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from .._weights.variants.matrix_mixture import MatrixWeightsMixture
        from .._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        if isinstance(bank_module, (MatrixWeightsMixture, MatrixBiasMixture)):
            # The observation already contains the selected probability mass.
            return _BankDistributionSummary(
                per_slot_utilization=bank_logits.mean(dim=0),
                mean_per_sample_entropy=self.__distribution_entropy(
                    bank_logits, -1
                ).mean(),
            )
        if isinstance(bank_module, SoftWeightedBankDynamicWeight):
            return self.__summarize_soft_weighted_bank(bank_module, bank_logits)
        if isinstance(bank_module, LayeredWeightedBankDynamicWeight):
            return self.__summarize_layered_weighted_bank(bank_module, bank_logits)
        if isinstance(bank_module, WeightedBankDynamicBias):
            return self.__summarize_weighted_bank_bias(bank_module, bank_logits)
        return None

    @staticmethod
    def __distribution_entropy(distribution: Tensor, dimension: int) -> Tensor:
        safe_distribution = distribution.clamp_min(1e-9)
        return -(safe_distribution.log() * distribution).sum(dim=dimension)

    @classmethod
    def __summarize_soft_weighted_bank(
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
            mean_per_sample_entropy=cls.__distribution_entropy(
                bank_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def __summarize_layered_weighted_bank(
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
            mean_per_sample_entropy=cls.__distribution_entropy(
                bank_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def __summarize_weighted_bank_bias(
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
            mean_per_sample_entropy=cls.__distribution_entropy(
                flat_distribution,
                dimension=-1,
            ).mean(),
        )

    @classmethod
    def __calculate_utilization(
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
            marginal_entropy=cls.__distribution_entropy(utilization, dimension=-1),
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
        self._method_restorers: list[Callable[[], None]] = []
        self._bank_modules: list[tuple[str, Module]] = []
        self._utilization_history: dict[str, MonitorTensorHistory] = {}
        self._last_bank_logits: dict[str, Tensor] = {}
        self._emission_policy = MonitorEmissionPolicy()
        self._diagnostics = _WeightBankDiagnostics()

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
            from .._biases.variants.matrix_mixture import MatrixBiasMixture
            from .._weights.variants.matrix_mixture import MatrixWeightsMixture

            if isinstance(bank_module, (MatrixWeightsMixture, MatrixBiasMixture)):
                reduction_method = (
                    "_MatrixWeightsMixture__reduce_mixture"
                    if isinstance(bank_module, MatrixWeightsMixture)
                    else "_MatrixBiasMixture__reduce_mixture"
                )
                self.__wrap_mixture_reduction(
                    module_name, bank_module, reduction_method
                )
                continue
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

        from .._biases.variants.matrix_mixture import MatrixBiasMixture
        from .._weights.variants.layered_weighted_bank import (
            LayeredWeightedBankDynamicWeight,
        )
        from .._weights.variants.matrix_mixture import MatrixWeightsMixture
        from .._weights.variants.soft_weighted_bank import (
            SoftWeightedBankDynamicWeight,
        )

        return isinstance(
            module,
            (
                MatrixWeightsMixture,
                MatrixBiasMixture,
                LayeredWeightedBankDynamicWeight,
                SoftWeightedBankDynamicWeight,
                WeightedBankDynamicBias,
            ),
        )

    def __wrap_mixture_reduction(
        self, module_name: str, bank_module: Module, method_name: str
    ) -> None:
        original_method = getattr(bank_module, method_name)
        had_instance_override = method_name in vars(bank_module)

        def monitored_reduction(probabilities, indices):
            output = original_method(probabilities, indices)
            self.__capture_selected_route(
                module_name, bank_module, probabilities, indices
            )
            return output

        def restore_method() -> None:
            if had_instance_override:
                setattr(bank_module, method_name, original_method)
            else:
                delattr(bank_module, method_name)

        setattr(bank_module, method_name, monitored_reduction)
        self._method_restorers.append(restore_method)

    def __capture_selected_route(
        self,
        module_name: str,
        bank_module: Module,
        probabilities: Tensor | None,
        indices: Tensor | None,
    ) -> None:
        if probabilities is None:
            return
        coefficients = probabilities.detach().reshape(-1, bank_module.top_k)
        if indices is None:
            distribution = coefficients
        else:
            distribution = coefficients.new_zeros(
                (coefficients.shape[0], bank_module.parameter_bank.size(0))
            )
            distribution.scatter_add_(
                1, indices.detach().reshape_as(coefficients), coefficients
            )
        self._last_bank_logits[module_name] = distribution

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
            if bank_logits.numel() == 0:
                continue
            diagnostic_facts = self._diagnostics.collect(
                bank_module,
                bank_logits,
                dead_slot_utilization_floor=self.DEAD_SLOT_UTILIZATION_FLOOR,
                include_per_slot_scalars=self.log_per_slot_scalars,
            )
            if diagnostic_facts is None:
                continue
            self.__emit_diagnostic_facts(
                pl_module,
                module_name,
                diagnostic_facts,
            )

    def __emit_diagnostic_facts(
        self,
        pl_module: LightningModule,
        module_name: str,
        diagnostic_facts: _WeightBankDiagnosticFacts,
    ) -> None:
        metric_prefix = f"{module_name}/bank"
        for metric in diagnostic_facts.scalars:
            pl_module.log(f"{metric_prefix}/{metric.suffix}", metric.value)
        experiment = getattr(pl_module.logger, "experiment", None)
        if experiment is None:
            return
        global_step = pl_module.global_step
        self._utilization_history[module_name].append(
            diagnostic_facts.per_slot_utilization
        )
        self._emission_policy.emit_histogram(
            experiment,
            f"{metric_prefix}/histogram/utilization",
            diagnostic_facts.per_slot_utilization,
            global_step,
        )
        self._emission_policy.emit_history_heatmap(
            experiment,
            f"{metric_prefix}/heatmap/utilization",
            self._utilization_history[module_name],
            global_step,
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
        for restore_method in reversed(self._method_restorers):
            restore_method()
        self._method_restorers.clear()
        self._bank_modules.clear()
        self._utilization_history.clear()
        self._last_bank_logits.clear()
        self._emission_policy.clear()
