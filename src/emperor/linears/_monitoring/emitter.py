from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from emperor.linears._monitoring.diagnostics import (
    _LinearActivationTrackingContext,
    _LinearParameterChannelMetrics,
    _LinearTrackingContext,
)

if TYPE_CHECKING:
    from lightning import LightningModule
    from torch import Tensor


def _log_metric(
    pl_module: LightningModule,
    name: str,
    value: Tensor,
) -> None:
    pl_module.log(name, value, sync_dist=True)


class _LinearMetricEmitter:
    """Own the Linear monitor's metric names, conditions, and emission order."""

    DEAD_FEATURE_RELATIVE_FLOOR = 1e-3

    def __init__(self, dead_feature_relative_floor: float) -> None:
        self.dead_feature_relative_floor = dead_feature_relative_floor

    def emit_activation(self, context: _LinearActivationTrackingContext) -> None:
        if context.input_summary is not None:
            self.__track_input_mean(context)
            self.__track_input_variance(context)
        if context.output_summary is not None:
            self.__track_output_mean(context)
            self.__track_output_variance(context)

    @staticmethod
    def __track_input_mean(context: _LinearActivationTrackingContext) -> None:
        assert context.input_summary is not None
        _log_metric(
            context.pl_module,
            f"{context.module_name}/input/mean",
            context.input_summary.mean,
        )

    @staticmethod
    def __track_input_variance(context: _LinearActivationTrackingContext) -> None:
        assert context.input_summary is not None
        _log_metric(
            context.pl_module,
            f"{context.module_name}/input/var",
            context.input_summary.variance,
        )

    @staticmethod
    def __track_output_mean(context: _LinearActivationTrackingContext) -> None:
        assert context.output_summary is not None
        _log_metric(
            context.pl_module,
            f"{context.module_name}/output/mean",
            context.output_summary.mean,
        )

    @staticmethod
    def __track_output_variance(context: _LinearActivationTrackingContext) -> None:
        assert context.output_summary is not None
        _log_metric(
            context.pl_module,
            f"{context.module_name}/output/var",
            context.output_summary.variance,
        )

    def emit_training(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        self.__track_parameter_mean(contexts, "weights")
        self.__track_parameter_variance(contexts, "weights")
        self.__track_parameter_l2_norm(contexts, "weights")
        self.__track_parameter_delta_norm(contexts, "weights")
        self.__track_relative_parameter_delta_norm(contexts, "weights")
        self.__track_parameter_mean(contexts, "bias")
        self.__track_parameter_variance(contexts, "bias")
        self.__track_parameter_l2_norm(contexts, "bias")
        self.__track_parameter_delta_norm(contexts, "bias")
        self.__track_relative_parameter_delta_norm(contexts, "bias")
        self.__track_gradient_mean(contexts, "weights")
        self.__track_gradient_variance(contexts, "weights")
        self.__track_gradient_norm(contexts, "weights")
        self.__track_gradient_to_weight_norm_ratio(contexts)
        self.__track_update_ratio(contexts)
        self.__track_gradient_mean(contexts, "bias")
        self.__track_gradient_variance(contexts, "bias")
        self.__track_gradient_norm(contexts, "bias")
        self.__track_dead_input_fraction(contexts)
        self.__track_dead_output_fraction(contexts)
        self.__track_spectral_norm(contexts)
        self.__track_condition_number(contexts)
        self.__track_effective_rank(contexts)

    @staticmethod
    def __channel_metrics(
        context: _LinearTrackingContext,
        parameter_channel: str,
    ) -> _LinearParameterChannelMetrics | None:
        return context.weights if parameter_channel == "weights" else context.bias

    def __track_parameter_mean(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/mean",
                    metrics.summary.mean,
                )

    def __track_parameter_variance(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/var",
                    metrics.summary.variance,
                )

    def __track_parameter_l2_norm(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/l2_norm",
                    metrics.summary.norm,
                )

    def __track_parameter_delta_norm(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.change is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/delta_norm",
                    metrics.change.delta_norm,
                )

    def __track_relative_parameter_delta_norm(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.change is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/relative_delta_norm",
                    metrics.change.relative_delta_norm,
                )

    def __track_gradient_mean(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.gradient_summary is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/grad_mean",
                    metrics.gradient_summary.mean,
                )

    def __track_gradient_variance(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.gradient_summary is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/grad_var",
                    metrics.gradient_summary.variance,
                )

    def __track_gradient_norm(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.gradient_summary is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/{parameter_channel}/grad_norm",
                    metrics.gradient_summary.norm,
                )

    @staticmethod
    def __track_gradient_to_weight_norm_ratio(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            ratio = context.weights.gradient_to_weight_norm_ratio
            if ratio is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/weights/gradient_to_weight_norm_ratio",
                    ratio,
                )

    @staticmethod
    def __track_update_ratio(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weights.update_ratio is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/weights/update_ratio",
                    context.weights.update_ratio,
                )

    def __track_dead_input_fraction(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            _log_metric(
                context.pl_module,
                f"{context.module_name}/weights/dead_input_fraction",
                self.__dead_feature_fraction(context.input_feature_norms),
            )

    def __track_dead_output_fraction(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            _log_metric(
                context.pl_module,
                f"{context.module_name}/weights/dead_output_fraction",
                self.__dead_feature_fraction(context.output_feature_norms),
            )

    def __dead_feature_fraction(self, feature_norms: Tensor) -> Tensor:
        if not torch.isfinite(feature_norms).all():
            return torch.full(
                (),
                float("nan"),
                dtype=feature_norms.dtype,
                device=feature_norms.device,
            )
        dead_threshold = self.dead_feature_relative_floor * feature_norms.mean()
        return (feature_norms <= dead_threshold).float().mean()

    @staticmethod
    def __track_spectral_norm(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/weights/spectral_norm",
                    context.weight_conditioning.spectral_norm,
                )

    @staticmethod
    def __track_condition_number(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/weights/condition_number",
                    context.weight_conditioning.condition_number,
                )

    @staticmethod
    def __track_effective_rank(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                _log_metric(
                    context.pl_module,
                    f"{context.module_name}/weights/effective_rank",
                    context.weight_conditioning.effective_rank,
                )
