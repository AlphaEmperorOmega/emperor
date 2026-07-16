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
class _InputAdaptivityMetrics:
    cross_sample_standard_deviation: Tensor
    adaptivity_ratio: Tensor
    centroid_cosine_mean: Tensor


@dataclass(frozen=True)
class _AdaptiveParameterTrackingContext:
    pl_module: LightningModule
    metric_prefix: str
    slot: AdaptiveParameterSlot
    option: Module
    observation: _AdaptiveParameterObservation
    input_adaptivity: _InputAdaptivityMetrics | None
    weight_bank_values: Tensor | None
    effective_scale: Tensor | None
    experiment: object | None
    global_step: int


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
            )

    def __attach_option_hooks(
        self,
        augmentation_path: str,
        augmentation: Module,
        pl_module: LightningModule,
    ) -> None:
        for attribute_name, metric_slot in self._OPTION_SLOTS:
            option = getattr(augmentation, attribute_name, None)
            if option is None:
                continue
            self._hooks.append(
                option.register_forward_hook(
                    self.__make_forward_hook(
                        augmentation_path,
                        metric_slot,
                        pl_module,
                    )
                )
            )

    def __make_forward_hook(
        self,
        augmentation_path: str,
        slot: AdaptiveParameterSlot,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_option_output(
            option: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            global_step = getattr(pl_module, "global_step", 0)
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
            )

        return log_option_output

    def __emit_observation(
        self,
        pl_module: LightningModule,
        augmentation_path: str,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
    ) -> None:
        context = self.__build_tracking_context(
            pl_module,
            augmentation_path,
            slot,
            option,
            observation,
        )
        self.__track_adaptive_parameter_diagnostics(context)

    def __build_tracking_context(
        self,
        pl_module: LightningModule,
        augmentation_path: str,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
    ) -> _AdaptiveParameterTrackingContext:
        return _AdaptiveParameterTrackingContext(
            pl_module=pl_module,
            metric_prefix=f"{augmentation_path}/{slot}/batch",
            slot=slot,
            option=option,
            observation=observation,
            input_adaptivity=self.__calculate_input_adaptivity(observation),
            weight_bank_values=self.__weight_bank_values(slot, option),
            effective_scale=self.__effective_bias_scale(slot, option, observation),
            experiment=getattr(
                getattr(pl_module, "logger", None),
                "experiment",
                None,
            ),
            global_step=getattr(pl_module, "global_step", 0),
        )

    def __track_adaptive_parameter_diagnostics(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        self.__track_output_mean(context)
        self.__track_output_variance(context)
        self.__track_output_minimum(context)
        self.__track_output_maximum(context)
        self.__track_output_l2_norm(context)
        self.__track_output_maximum_absolute_value(context)
        self.__track_base_mean(context)
        self.__track_base_variance(context)
        self.__track_delta_mean(context)
        self.__track_delta_variance(context)
        self.__track_delta_l2_norm(context)
        self.__track_relative_delta_norm(context)
        self.__track_cross_sample_standard_deviation(context)
        self.__track_adaptivity_ratio(context)
        self.__track_centroid_cosine_mean(context)
        self.__track_decay_step(context)
        self.__track_warmup_step(context)
        self.__track_scale(context)
        self.__track_clamp_limit(context)
        self.__track_weight_bank_mean(context)
        self.__track_weight_bank_variance(context)
        self.__track_weight_bank_l2_norm(context)
        self.__track_effective_scale_mean(context)
        self.__track_effective_scale_variance(context)
        self.__track_mask_relative_output_norm(context)
        self.__track_mask_attenuated_fraction(context)
        self.__track_mask_near_zero_fraction(context)
        self.__track_output_histogram(context)
        self.__track_delta_histogram(context)

    @staticmethod
    def __track_output_mean(context: _AdaptiveParameterTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_mean",
            context.observation.output.float().mean(),
        )

    @staticmethod
    def __track_output_variance(context: _AdaptiveParameterTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_var",
            context.observation.output.float().var(unbiased=False),
        )

    @staticmethod
    def __track_output_minimum(context: _AdaptiveParameterTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_min",
            context.observation.output.float().min(),
        )

    @staticmethod
    def __track_output_maximum(context: _AdaptiveParameterTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_max",
            context.observation.output.float().max(),
        )

    @staticmethod
    def __track_output_l2_norm(context: _AdaptiveParameterTrackingContext) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_l2_norm",
            context.observation.output.float().norm(),
        )

    @staticmethod
    def __track_output_maximum_absolute_value(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.metric_prefix}/output_max_abs",
            context.observation.output.float().abs().max(),
        )

    @staticmethod
    def __track_base_mean(context: _AdaptiveParameterTrackingContext) -> None:
        if context.observation.base is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/base_mean",
            context.observation.base.float().mean(),
        )

    @staticmethod
    def __track_base_variance(context: _AdaptiveParameterTrackingContext) -> None:
        if context.observation.base is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/base_var",
            context.observation.base.float().var(unbiased=False),
        )

    @staticmethod
    def __track_delta_mean(context: _AdaptiveParameterTrackingContext) -> None:
        if context.observation.delta is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/delta_mean",
            context.observation.delta.float().mean(),
        )

    @staticmethod
    def __track_delta_variance(context: _AdaptiveParameterTrackingContext) -> None:
        if context.observation.delta is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/delta_var",
            context.observation.delta.float().var(unbiased=False),
        )

    @staticmethod
    def __track_delta_l2_norm(context: _AdaptiveParameterTrackingContext) -> None:
        if context.observation.delta is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/delta_l2_norm",
            context.observation.delta.float().norm(),
        )

    @staticmethod
    def __track_relative_delta_norm(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.observation.base is None or context.observation.delta is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/relative_delta_norm",
            context.observation.delta.float().norm()
            / context.observation.base.float().norm().clamp_min(1e-6),
        )

    @staticmethod
    def __calculate_input_adaptivity(
        observation: _AdaptiveParameterObservation,
    ) -> _InputAdaptivityMetrics | None:
        adaptivity_values = (
            observation.delta if observation.delta is not None else observation.output
        )
        if adaptivity_values.dim() == 0 or adaptivity_values.shape[0] < 2:
            return None
        batch_size = adaptivity_values.shape[0]
        per_sample_values = adaptivity_values.float().reshape(batch_size, -1)
        centroid = per_sample_values.mean(dim=0)
        centered_values = per_sample_values - centroid
        return _InputAdaptivityMetrics(
            cross_sample_standard_deviation=centered_values.pow(2).mean().sqrt(),
            adaptivity_ratio=(
                centered_values.norm() / per_sample_values.norm().clamp_min(1e-12)
            ),
            centroid_cosine_mean=(
                AdaptiveParameterMonitorCallback.__mean_cosine_to_centroid(
                    per_sample_values,
                    centroid,
                )
            ),
        )

    @staticmethod
    def __track_cross_sample_standard_deviation(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.input_adaptivity is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/cross_sample_std",
            context.input_adaptivity.cross_sample_standard_deviation,
        )

    @staticmethod
    def __track_adaptivity_ratio(context: _AdaptiveParameterTrackingContext) -> None:
        if context.input_adaptivity is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/adaptivity_ratio",
            context.input_adaptivity.adaptivity_ratio,
        )

    @staticmethod
    def __track_centroid_cosine_mean(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.input_adaptivity is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/centroid_cosine_mean",
            context.input_adaptivity.centroid_cosine_mean,
        )

    @staticmethod
    def __mean_cosine_to_centroid(
        per_sample_values: Tensor,
        centroid: Tensor,
    ) -> Tensor:
        normalized_samples = F.normalize(per_sample_values, dim=1)
        normalized_centroid = F.normalize(centroid, dim=0)
        return (normalized_samples @ normalized_centroid).mean()

    def __track_decay_step(self, context: _AdaptiveParameterTrackingContext) -> None:
        self.__track_weight_internal_value(context, "decay_step")

    def __track_warmup_step(self, context: _AdaptiveParameterTrackingContext) -> None:
        self.__track_weight_internal_value(context, "warmup_step")

    def __track_scale(self, context: _AdaptiveParameterTrackingContext) -> None:
        self.__track_weight_internal_value(context, "scale")

    def __track_clamp_limit(self, context: _AdaptiveParameterTrackingContext) -> None:
        self.__track_weight_internal_value(context, "clamp_limit")

    def __track_weight_internal_value(
        self,
        context: _AdaptiveParameterTrackingContext,
        attribute_name: str,
    ) -> None:
        if not self.log_internal_stats or context.slot != "weight":
            return
        value = getattr(context.option, attribute_name, None)
        if torch.is_tensor(value):
            context.pl_module.log(
                f"{context.metric_prefix}/{attribute_name}",
                value.detach().float().mean(),
            )

    def __weight_bank_values(
        self,
        slot: AdaptiveParameterSlot,
        option: Module,
    ) -> Tensor | None:
        if not self.log_internal_stats or slot not in ("weight", "bias"):
            return None
        weight_bank = getattr(option, "weight_bank", None)
        return weight_bank.detach().float() if torch.is_tensor(weight_bank) else None

    def __effective_bias_scale(
        self,
        slot: AdaptiveParameterSlot,
        option: Module,
        observation: _AdaptiveParameterObservation,
    ) -> Tensor | None:
        if (
            not self.log_internal_stats
            or slot != "bias"
            or observation.base is None
            or not self.__uses_multiplicative_bias_scale(option)
        ):
            return None
        base_values = observation.base.float()
        if torch.any(base_values.abs() <= 1e-6):
            return None
        return observation.output.float() / base_values

    @staticmethod
    def __uses_multiplicative_bias_scale(option: Module) -> bool:
        from emperor.augmentations.adaptive_parameters._biases.affine import (
            AffineTransformDynamicBias,
        )
        from emperor.augmentations.adaptive_parameters._biases.gated import (
            SigmoidGatedDynamicBias,
            TanhGatedDynamicBias,
        )
        from emperor.augmentations.adaptive_parameters._biases.multiplicative import (
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
    def __track_weight_bank_mean(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.weight_bank_values is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/weight_bank_mean",
            context.weight_bank_values.mean(),
        )

    @staticmethod
    def __track_weight_bank_variance(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.weight_bank_values is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/weight_bank_var",
            context.weight_bank_values.var(unbiased=False),
        )

    @staticmethod
    def __track_weight_bank_l2_norm(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.weight_bank_values is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/weight_bank_l2_norm",
            context.weight_bank_values.norm(),
        )

    @staticmethod
    def __track_effective_scale_mean(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.effective_scale is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/effective_scale_mean",
            context.effective_scale.mean(),
        )

    @staticmethod
    def __track_effective_scale_variance(
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if context.effective_scale is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/effective_scale_var",
            context.effective_scale.var(unbiased=False),
        )

    @staticmethod
    def __can_track_mask(context: _AdaptiveParameterTrackingContext) -> bool:
        return context.slot == "mask" and context.observation.base is not None

    def __track_mask_relative_output_norm(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if not self.log_internal_stats or not self.__can_track_mask(context):
            return
        context.pl_module.log(
            f"{context.metric_prefix}/relative_output_norm",
            context.observation.output.float().norm()
            / context.observation.base.float().norm().clamp_min(1e-6),
        )

    def __track_mask_attenuated_fraction(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if not self.log_internal_stats or not self.__can_track_mask(context):
            return
        context.pl_module.log(
            f"{context.metric_prefix}/attenuated_fraction",
            (
                context.observation.output.float().abs()
                < context.observation.base.float().abs()
            )
            .float()
            .mean(),
        )

    def __track_mask_near_zero_fraction(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if not self.log_internal_stats or not self.__can_track_mask(context):
            return
        context.pl_module.log(
            f"{context.metric_prefix}/near_zero_fraction",
            (context.observation.output.float().abs() <= 1e-6).float().mean(),
        )

    def __track_output_histogram(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if not self.log_histograms or context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.metric_prefix}/output",
            context.observation.output,
            context.global_step,
        )

    def __track_delta_histogram(
        self,
        context: _AdaptiveParameterTrackingContext,
    ) -> None:
        if (
            not self.log_histograms
            or context.experiment is None
            or context.observation.delta is None
        ):
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.metric_prefix}/delta",
            context.observation.delta,
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
        self._emission_policy.clear()
