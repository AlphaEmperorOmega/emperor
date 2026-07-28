from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from emperor.monitoring import MonitorEmissionPolicy

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


AdaptiveParameterSlot = Literal["weight", "diagonal", "bias", "mask"]


@dataclass(frozen=True)
class _AdaptiveParameterObservation:
    output: Tensor
    base: Tensor | None
    delta: Tensor | None

    @classmethod
    def from_forward(
        cls,
        inputs: tuple[object, ...],
        output: Tensor,
    ) -> _AdaptiveParameterObservation:
        detached_output = output.detach()
        base = inputs[0].detach() if inputs and torch.is_tensor(inputs[0]) else None
        delta = detached_output - base if base is not None else None
        return cls(output=detached_output, base=base, delta=delta)


@dataclass(frozen=True)
class _AdaptiveParameterMetric:
    suffix: str
    value: Tensor


@dataclass(frozen=True)
class _AdaptiveParameterDiagnosticFacts:
    scalars: tuple[_AdaptiveParameterMetric, ...]
    histograms: tuple[_AdaptiveParameterMetric, ...]


class _AdaptiveParameterDiagnostics:
    """Collect ordered adaptive-parameter facts without delivery concerns."""

    __slots__ = ()

    def collect(
        self,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
        *,
        include_internal_stats: bool,
        include_histograms: bool,
        suppress_input_adaptivity: bool = False,
    ) -> _AdaptiveParameterDiagnosticFacts:
        scalar_metrics = [
            *self.__collect_output_metrics(observation),
            *self.__collect_base_and_delta_metrics(observation),
        ]
        if not suppress_input_adaptivity:
            scalar_metrics.extend(self.__collect_input_adaptivity_metrics(observation))
        if include_internal_stats:
            scalar_metrics.extend(
                self.__collect_internal_metrics(slot, option, observation)
            )
        histogram_metrics = (
            self.__collect_histogram_metrics(observation)
            if include_histograms
            else ()
        )
        return _AdaptiveParameterDiagnosticFacts(
            scalars=tuple(scalar_metrics),
            histograms=histogram_metrics,
        )

    @staticmethod
    def __collect_output_metrics(
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        output = observation.output.float()
        return (
            _AdaptiveParameterMetric("output_mean", output.mean()),
            _AdaptiveParameterMetric("output_var", output.var(unbiased=False)),
            _AdaptiveParameterMetric("output_min", output.min()),
            _AdaptiveParameterMetric("output_max", output.max()),
            _AdaptiveParameterMetric("output_l2_norm", output.norm()),
            _AdaptiveParameterMetric("output_max_abs", output.abs().max()),
        )

    @staticmethod
    def __collect_base_and_delta_metrics(
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        metrics: list[_AdaptiveParameterMetric] = []
        base = observation.base.float() if observation.base is not None else None
        delta = observation.delta.float() if observation.delta is not None else None
        if base is not None:
            metrics.extend(
                (
                    _AdaptiveParameterMetric("base_mean", base.mean()),
                    _AdaptiveParameterMetric("base_var", base.var(unbiased=False)),
                )
            )
        if delta is not None:
            metrics.extend(
                (
                    _AdaptiveParameterMetric("delta_mean", delta.mean()),
                    _AdaptiveParameterMetric("delta_var", delta.var(unbiased=False)),
                    _AdaptiveParameterMetric("delta_l2_norm", delta.norm()),
                )
            )
        if base is not None and delta is not None:
            metrics.append(
                _AdaptiveParameterMetric(
                    "relative_delta_norm",
                    delta.norm() / base.norm().clamp_min(1e-6),
                )
            )
        return tuple(metrics)

    def __collect_input_adaptivity_metrics(
        self,
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        adaptivity_values = (
            observation.delta if observation.delta is not None else observation.output
        )
        if adaptivity_values.dim() == 0 or adaptivity_values.shape[0] < 2:
            return ()
        batch_size = adaptivity_values.shape[0]
        per_sample_values = adaptivity_values.float().reshape(batch_size, -1)
        centroid = per_sample_values.mean(dim=0)
        centered_values = per_sample_values - centroid
        return (
            _AdaptiveParameterMetric(
                "cross_sample_std",
                centered_values.pow(2).mean().sqrt(),
            ),
            _AdaptiveParameterMetric(
                "adaptivity_ratio",
                centered_values.norm() / per_sample_values.norm().clamp_min(1e-12),
            ),
            _AdaptiveParameterMetric(
                "centroid_cosine_mean",
                self.__mean_cosine_to_centroid(per_sample_values, centroid),
            ),
        )

    @staticmethod
    def __mean_cosine_to_centroid(
        per_sample_values: Tensor,
        centroid: Tensor,
    ) -> Tensor:
        normalized_samples = F.normalize(per_sample_values, dim=1)
        normalized_centroid = F.normalize(centroid, dim=0)
        return (normalized_samples @ normalized_centroid).mean()

    def __collect_internal_metrics(
        self,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        return (
            *self.__collect_weight_internal_metrics(slot, option),
            *self.__collect_weight_bank_metrics(slot, option),
            *self.__collect_effective_scale_metrics(slot, option, observation),
            *self.__collect_mask_metrics(slot, observation),
        )

    @staticmethod
    def __collect_weight_internal_metrics(
        slot: AdaptiveParameterSlot,
        option: Module,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        if slot != "weight":
            return ()
        metrics: list[_AdaptiveParameterMetric] = []
        for attribute_name in (
            "decay_step",
            "warmup_step",
            "scale",
            "clamp_limit",
        ):
            value = getattr(option, attribute_name, None)
            if torch.is_tensor(value):
                metrics.append(
                    _AdaptiveParameterMetric(
                        attribute_name,
                        value.detach().float().mean(),
                    )
                )
        return tuple(metrics)

    @staticmethod
    def __collect_weight_bank_metrics(
        slot: AdaptiveParameterSlot,
        option: Module,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        if slot not in ("weight", "bias"):
            return ()
        weight_bank = getattr(option, "weight_bank", None)
        if not torch.is_tensor(weight_bank):
            return ()
        weight_bank_values = weight_bank.detach().float()
        return (
            _AdaptiveParameterMetric("weight_bank_mean", weight_bank_values.mean()),
            _AdaptiveParameterMetric(
                "weight_bank_var",
                weight_bank_values.var(unbiased=False),
            ),
            _AdaptiveParameterMetric(
                "weight_bank_l2_norm",
                weight_bank_values.norm(),
            ),
        )

    def __collect_effective_scale_metrics(
        self,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        if (
            slot != "bias"
            or observation.base is None
            or not self.__uses_multiplicative_bias_scale(option)
        ):
            return ()
        base_values = observation.base.float()
        if torch.any(base_values.abs() <= 1e-6):
            return ()
        effective_scale = observation.output.float() / base_values
        return (
            _AdaptiveParameterMetric("effective_scale_mean", effective_scale.mean()),
            _AdaptiveParameterMetric(
                "effective_scale_var",
                effective_scale.var(unbiased=False),
            ),
        )

    @staticmethod
    def __uses_multiplicative_bias_scale(option: Module) -> bool:
        from emperor.augmentations.adaptive_parameters._biases.variants.affine import (
            AffineTransformDynamicBias,
        )
        from emperor.augmentations.adaptive_parameters._biases.variants.gated import (
            SigmoidGatedDynamicBias,
            TanhGatedDynamicBias,
        )
        from emperor.augmentations.adaptive_parameters._biases.variants.multiplicative import (
            MultiplicativeDynamicBias,
        )

        return isinstance(
            option,
            (
                AffineTransformDynamicBias,
                MultiplicativeDynamicBias,
                SigmoidGatedDynamicBias,
                TanhGatedDynamicBias,
            ),
        )

    @staticmethod
    def __collect_mask_metrics(
        slot: AdaptiveParameterSlot,
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        if slot != "mask" or observation.base is None:
            return ()
        output = observation.output.float()
        base = observation.base.float()
        return (
            _AdaptiveParameterMetric(
                "relative_output_norm",
                output.norm() / base.norm().clamp_min(1e-6),
            ),
            _AdaptiveParameterMetric(
                "attenuated_fraction",
                (output.abs() < base.abs()).float().mean(),
            ),
            _AdaptiveParameterMetric(
                "near_zero_fraction",
                (output.abs() <= 1e-6).float().mean(),
            ),
        )

    @staticmethod
    def __collect_histogram_metrics(
        observation: _AdaptiveParameterObservation,
    ) -> tuple[_AdaptiveParameterMetric, ...]:
        metrics = [_AdaptiveParameterMetric("output", observation.output)]
        if observation.delta is not None:
            metrics.append(_AdaptiveParameterMetric("delta", observation.delta))
        return tuple(metrics)


class AdaptiveParameterMonitorCallback(Callback):
    """Log batch diagnostics for enabled adaptive-parameter slots."""

    _OPTION_SLOTS: tuple[tuple[str, AdaptiveParameterSlot], ...] = (
        ("weight_model", "weight"),
        ("diagonal_model", "diagonal"),
        ("bias_model", "bias"),
        ("mask_model", "mask"),
    )

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_histograms: bool = False,
        log_internal_stats: bool = True,
    ) -> None:
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.log_histograms = log_histograms
        self.log_internal_stats = log_internal_stats
        self._hooks: list[RemovableHandle] = []
        self._emission_policy = MonitorEmissionPolicy()
        self._diagnostics = _AdaptiveParameterDiagnostics()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.augmentations.adaptive_parameters._augmentation import (
            AdaptiveParameterAugmentation,
        )

        self.__cleanup()
        for augmentation_path, augmentation in pl_module.named_modules():
            if not isinstance(augmentation, AdaptiveParameterAugmentation):
                continue
            self.__attach_option_hooks(
                augmentation_path,
                augmentation,
                pl_module,
                suppress_input_adaptivity=(
                    augmentation.adaptive_parameter_grouping_enabled
                ),
            )

    def __attach_option_hooks(
        self,
        augmentation_path: str,
        augmentation: Module,
        pl_module: LightningModule,
        *,
        suppress_input_adaptivity: bool,
    ) -> None:
        for attribute_name, metric_slot in self._OPTION_SLOTS:
            option = getattr(augmentation, attribute_name)
            if option is None:
                continue
            self._hooks.append(
                option.register_forward_hook(
                    self.__make_forward_hook(
                        augmentation_path,
                        metric_slot,
                        pl_module,
                        suppress_input_adaptivity=suppress_input_adaptivity,
                    )
                )
            )

    def __make_forward_hook(
        self,
        augmentation_path: str,
        slot: AdaptiveParameterSlot,
        pl_module: LightningModule,
        *,
        suppress_input_adaptivity: bool,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_option_output(
            option: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            global_step = pl_module.global_step
            if global_step % self.log_every_n_steps != 0:
                return
            if not torch.is_tensor(output):
                return
            observation = _AdaptiveParameterObservation.from_forward(inputs, output)
            self.__emit_observation(
                pl_module,
                augmentation_path,
                slot,
                option,
                observation,
                suppress_input_adaptivity=suppress_input_adaptivity,
            )

        return log_option_output

    def __emit_observation(
        self,
        pl_module: LightningModule,
        augmentation_path: str,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
        *,
        suppress_input_adaptivity: bool,
    ) -> None:
        metric_prefix = f"{augmentation_path}/{slot}/batch"
        experiment = getattr(pl_module.logger, "experiment", None)
        diagnostic_facts = self._diagnostics.collect(
            slot,
            option,
            observation,
            include_internal_stats=self.log_internal_stats,
            include_histograms=self.log_histograms and experiment is not None,
            suppress_input_adaptivity=suppress_input_adaptivity,
        )
        for metric in diagnostic_facts.scalars:
            pl_module.log(f"{metric_prefix}/{metric.suffix}", metric.value)
        if experiment is None:
            return
        global_step = pl_module.global_step
        for histogram in diagnostic_facts.histograms:
            self._emission_policy.emit_histogram(
                experiment,
                f"{metric_prefix}/{histogram.suffix}",
                histogram.value,
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
        self._emission_policy.clear()
