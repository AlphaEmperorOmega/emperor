from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.linears._layer import LinearAbstract
from emperor.linears._monitoring.diagnostics import (
    _LinearDiagnostics,
    _LinearParameterChannelMetrics,
    _LinearTrackingContext,
    _ParameterChangeMetrics,
    _TensorSummary,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module, Parameter
    from torch.utils.hooks import RemovableHandle


class LinearMonitorCallback(Callback):
    """Log activations, parameters, gradients, and matrix health for linear layers."""

    DEAD_FEATURE_RELATIVE_FLOOR = 1e-3

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_weight_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(log_every_n_steps, bool) or not isinstance(
            log_every_n_steps, int
        ):
            raise TypeError("log_every_n_steps must be an int.")
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        if not isinstance(log_weight_conditioning, bool):
            raise TypeError("log_weight_conditioning must be a bool.")
        self.log_every_n_steps = log_every_n_steps
        self.log_weight_conditioning = log_weight_conditioning
        self._hooks: list[RemovableHandle] = []
        self._linear_modules: list[tuple[str, LinearAbstract]] = []
        self._parameter_snapshots: dict[tuple[str, str], Tensor] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()
        for module_name, linear_layer in pl_module.named_modules():
            if not isinstance(linear_layer, LinearAbstract):
                continue
            self._linear_modules.append((module_name, linear_layer))
            self._hooks.append(
                linear_layer.register_forward_hook(
                    self.__make_forward_stats_hook(module_name, pl_module)
                )
            )

    def __make_forward_stats_hook(
        self,
        module_name: str,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_forward_stats(
            _linear_layer: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if pl_module.global_step % self.log_every_n_steps != 0:
                return
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                return
            input_summary = _LinearDiagnostics.summarize(inputs[0])
            output_summary = _LinearDiagnostics.summarize(output)
            self.__track_forward_diagnostics(
                pl_module,
                module_name,
                input_summary,
                output_summary,
            )

        return log_forward_stats

    def __track_forward_diagnostics(
        self,
        pl_module: LightningModule,
        module_name: str,
        input_summary: _TensorSummary,
        output_summary: _TensorSummary,
    ) -> None:
        self.__track_input_mean(pl_module, module_name, input_summary)
        self.__track_input_variance(pl_module, module_name, input_summary)
        self.__track_output_mean(pl_module, module_name, output_summary)
        self.__track_output_variance(pl_module, module_name, output_summary)

    @staticmethod
    def __track_input_mean(
        pl_module: LightningModule,
        module_name: str,
        input_summary: _TensorSummary,
    ) -> None:
        pl_module.log(f"{module_name}/input/mean", input_summary.mean)

    @staticmethod
    def __track_input_variance(
        pl_module: LightningModule,
        module_name: str,
        input_summary: _TensorSummary,
    ) -> None:
        pl_module.log(f"{module_name}/input/var", input_summary.variance)

    @staticmethod
    def __track_output_mean(
        pl_module: LightningModule,
        module_name: str,
        output_summary: _TensorSummary,
    ) -> None:
        pl_module.log(f"{module_name}/output/mean", output_summary.mean)

    @staticmethod
    def __track_output_variance(
        pl_module: LightningModule,
        module_name: str,
        output_summary: _TensorSummary,
    ) -> None:
        pl_module.log(f"{module_name}/output/var", output_summary.variance)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        contexts = self.__build_tracking_contexts(pl_module)
        self.__track_linear_training_diagnostics(contexts)

    def __build_tracking_contexts(
        self,
        pl_module: LightningModule,
    ) -> tuple[_LinearTrackingContext, ...]:
        contexts = []
        for module_name, linear_layer in self._linear_modules:
            weight_values = linear_layer.weight_params.detach().float()
            weights = self.__build_parameter_channel_metrics(
                module_name,
                "weights",
                linear_layer.weight_params,
                include_update_ratio=True,
            )
            bias = None
            if linear_layer.bias_params is None:
                self._parameter_snapshots.pop((module_name, "bias"), None)
            else:
                bias = self.__build_parameter_channel_metrics(
                    module_name,
                    "bias",
                    linear_layer.bias_params,
                    include_update_ratio=False,
                )
            contexts.append(
                _LinearTrackingContext(
                    pl_module=pl_module,
                    module_name=module_name,
                    weights=weights,
                    bias=bias,
                    input_feature_norms=weight_values.norm(dim=1),
                    output_feature_norms=weight_values.norm(dim=0),
                    weight_conditioning=(
                        _LinearDiagnostics.weight_conditioning(weight_values)
                        if self.log_weight_conditioning
                        else None
                    ),
                )
            )
        return tuple(contexts)

    def __build_parameter_channel_metrics(
        self,
        module_name: str,
        parameter_channel: str,
        parameter: Parameter,
        *,
        include_update_ratio: bool,
    ) -> _LinearParameterChannelMetrics:
        current_values = parameter.detach().float()
        snapshot_key = (module_name, parameter_channel)
        previous_values = self._parameter_snapshots.get(snapshot_key)
        change = None
        if (
            previous_values is not None
            and previous_values.shape == current_values.shape
        ):
            delta_norm = (current_values - previous_values).norm()
            change = _ParameterChangeMetrics(
                delta_norm=delta_norm,
                relative_delta_norm=(
                    delta_norm / previous_values.norm().clamp_min(1e-12)
                ),
            )
        gradient_summary = (
            _LinearDiagnostics.summarize(parameter.grad)
            if parameter.grad is not None
            else None
        )
        return _LinearParameterChannelMetrics(
            values=current_values,
            summary=_LinearDiagnostics.summarize(current_values),
            change=change,
            gradient_summary=gradient_summary,
            update_ratio=(
                gradient_summary.norm / current_values.norm().clamp_min(1e-6)
                if include_update_ratio and gradient_summary is not None
                else None
            ),
        )

    def __track_linear_training_diagnostics(
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
        self.__record_parameter_snapshots(contexts)
        self.__track_gradient_mean(contexts, "weights")
        self.__track_gradient_variance(contexts, "weights")
        self.__track_gradient_norm(contexts, "weights")
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
                context.pl_module.log(
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
                context.pl_module.log(
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
                context.pl_module.log(
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
                context.pl_module.log(
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
                context.pl_module.log(
                    f"{context.module_name}/{parameter_channel}/relative_delta_norm",
                    metrics.change.relative_delta_norm,
                )

    def __record_parameter_snapshots(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            self._parameter_snapshots[(context.module_name, "weights")] = (
                context.weights.values.detach().clone()
            )
            if context.bias is not None:
                self._parameter_snapshots[(context.module_name, "bias")] = (
                    context.bias.values.detach().clone()
                )

    def __track_gradient_mean(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
        parameter_channel: str,
    ) -> None:
        for context in contexts:
            metrics = self.__channel_metrics(context, parameter_channel)
            if metrics is not None and metrics.gradient_summary is not None:
                context.pl_module.log(
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
                context.pl_module.log(
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
                context.pl_module.log(
                    f"{context.module_name}/{parameter_channel}/grad_norm",
                    metrics.gradient_summary.norm,
                )

    @staticmethod
    def __track_update_ratio(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weights.update_ratio is not None:
                context.pl_module.log(
                    f"{context.module_name}/weights/update_ratio",
                    context.weights.update_ratio,
                )

    def __track_dead_input_fraction(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            context.pl_module.log(
                f"{context.module_name}/weights/dead_input_fraction",
                self.__dead_feature_fraction(context.input_feature_norms),
            )

    def __track_dead_output_fraction(
        self,
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            context.pl_module.log(
                f"{context.module_name}/weights/dead_output_fraction",
                self.__dead_feature_fraction(context.output_feature_norms),
            )

    def __dead_feature_fraction(self, feature_norms: Tensor) -> Tensor:
        dead_threshold = self.DEAD_FEATURE_RELATIVE_FLOOR * feature_norms.mean()
        return (feature_norms <= dead_threshold).float().mean()

    @staticmethod
    def __track_spectral_norm(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                context.pl_module.log(
                    f"{context.module_name}/weights/spectral_norm",
                    context.weight_conditioning.spectral_norm,
                )

    @staticmethod
    def __track_condition_number(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                context.pl_module.log(
                    f"{context.module_name}/weights/condition_number",
                    context.weight_conditioning.condition_number,
                )

    @staticmethod
    def __track_effective_rank(
        contexts: tuple[_LinearTrackingContext, ...],
    ) -> None:
        for context in contexts:
            if context.weight_conditioning is not None:
                context.pl_module.log(
                    f"{context.module_name}/weights/effective_rank",
                    context.weight_conditioning.effective_rank,
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
        self._linear_modules.clear()
        self._parameter_snapshots.clear()
