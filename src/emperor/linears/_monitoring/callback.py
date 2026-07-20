from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback
from torch.nn.modules.module import register_module_forward_pre_hook

from emperor.linears._layer import LinearAbstract
from emperor.linears._monitoring.diagnostics import (
    _LinearDiagnostics,
    _LinearParameterChannelMetrics,
    _LinearTrackingContext,
    _ParameterChangeMetrics,
    _TensorMoments,
    _TensorSummary,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module, Parameter
    from torch.optim import Optimizer
    from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class _ParameterBeforeStep:
    parameter: Parameter
    values: Tensor
    gradient_summary: _TensorSummary | None


@dataclass(frozen=True)
class _LinearBeforeStep:
    module_name: str
    linear_layer: LinearAbstract
    weights: _ParameterBeforeStep
    bias: _ParameterBeforeStep | None


@dataclass(frozen=True)
class _PendingOptimizerStep:
    step: int
    linear_states: tuple[_LinearBeforeStep, ...]


class LinearMonitorCallback(Callback):
    """Log activations, parameters, gradients, and matrix health for linear layers."""

    DEAD_FEATURE_RELATIVE_FLOOR = 1e-3

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_weight_conditioning: bool = False,
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
        self._hooks: dict[str, RemovableHandle] = {}
        self._linear_modules: dict[str, LinearAbstract] = {}
        self._linear_module_names: dict[int, str] = {}
        self._activation_moments: dict[tuple[int, int, str], _TensorMoments] = {}
        self._activation_modules: dict[tuple[int, int], tuple[str, LinearAbstract]] = {}
        self._pending_step: _PendingOptimizerStep | None = None
        self._discovery_hook: RemovableHandle | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()
        self.__refresh_linear_modules(trainer, pl_module)

    def __make_linear_discovery_hook(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...]], None]:
        def discover_linear_module(
            candidate: Module,
            _inputs: tuple[object, ...],
        ) -> None:
            if not isinstance(candidate, LinearAbstract):
                return
            tracked_name = self._linear_module_names.get(id(candidate))
            if tracked_name is not None:
                try:
                    still_tracked = pl_module.get_submodule(tracked_name) is candidate
                except AttributeError:
                    still_tracked = False
                if still_tracked:
                    return
            self.__refresh_linear_modules(trainer, pl_module)

        return discover_linear_module

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        self.__refresh_linear_modules(trainer, pl_module)
        self.__remove_discovery_hook()
        target_step = int(trainer.global_step) + 1
        if target_step % self.log_every_n_steps == 0:
            self.__install_discovery_hook(trainer, pl_module)

    def __refresh_linear_modules(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        current_modules = {
            module_name: linear_layer
            for module_name, linear_layer in pl_module.named_modules()
            if isinstance(linear_layer, LinearAbstract)
        }
        for module_name, tracked_layer in tuple(self._linear_modules.items()):
            if current_modules.get(module_name) is tracked_layer:
                continue
            self._hooks.pop(module_name).remove()
            self._linear_modules.pop(module_name)

        for module_name, linear_layer in current_modules.items():
            if self._linear_modules.get(module_name) is linear_layer:
                continue
            self._linear_modules[module_name] = linear_layer
            self._hooks[module_name] = linear_layer.register_forward_hook(
                self.__make_forward_stats_hook(trainer, module_name, linear_layer),
                with_kwargs=True,
            )
        self._linear_module_names = {
            id(linear_layer): module_name
            for module_name, linear_layer in current_modules.items()
        }

    def __make_forward_stats_hook(
        self,
        trainer: Trainer,
        module_name: str,
        linear_layer: LinearAbstract,
    ) -> Callable[[Module, tuple[object, ...], dict[str, object], object], None]:
        def collect_forward_stats(
            _linear_layer: Module,
            inputs: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            if not trainer.training:
                return
            target_step = int(trainer.global_step) + 1
            if target_step % self.log_every_n_steps != 0:
                return
            module_key = id(linear_layer)
            self._activation_modules[(target_step, module_key)] = (
                module_name,
                linear_layer,
            )
            input_tensor = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if input_tensor is None and torch.is_tensor(kwargs.get("X")):
                input_tensor = kwargs["X"]
            if input_tensor is not None:
                self.__activation_moments(target_step, module_key, "input").add(
                    input_tensor
                )
            if torch.is_tensor(output):
                self.__activation_moments(target_step, module_key, "output").add(output)

        return collect_forward_stats

    def __activation_moments(
        self,
        step: int,
        module_key: int,
        channel: str,
    ) -> _TensorMoments:
        key = (step, module_key, channel)
        return self._activation_moments.setdefault(key, _TensorMoments())

    def __track_forward_diagnostics(
        self,
        pl_module: LightningModule,
        module_name: str,
        input_summary: _TensorSummary | None,
        output_summary: _TensorSummary | None,
    ) -> None:
        if input_summary is not None:
            self.__track_input_mean(pl_module, module_name, input_summary)
            self.__track_input_variance(pl_module, module_name, input_summary)
        if output_summary is not None:
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
        try:
            self.__finish_pending_step(trainer, pl_module)
        finally:
            self.__remove_discovery_hook()

    def __finish_pending_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> bool:
        pending_step = self._pending_step
        if pending_step is None or int(trainer.global_step) != pending_step.step:
            return False
        try:
            self.__track_buffered_activations(pl_module, pending_step)
            contexts = self.__build_tracking_contexts(pl_module, pending_step)
            self.__track_linear_training_diagnostics(contexts)
        finally:
            self.__discard_activations_through(pending_step.step)
            self._pending_step = None
            self.__refresh_linear_modules(trainer, pl_module)
        return True

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        target_step = int(trainer.global_step) + 1
        if self._pending_step is not None:
            if self._pending_step.step == target_step:
                return
            if not self.__finish_pending_step(trainer, pl_module):
                self.__discard_activations_through(self._pending_step.step)
                self._pending_step = None
        if target_step % self.log_every_n_steps != 0:
            if (target_step + 1) % self.log_every_n_steps == 0:
                self.__install_discovery_hook(trainer, pl_module)
            return
        self.__refresh_linear_modules(trainer, pl_module)
        optimizer_parameter_ids = self.__optimizer_parameter_ids(optimizer)
        linear_modules = self.__linear_modules_for_step(target_step)
        self._pending_step = _PendingOptimizerStep(
            step=target_step,
            linear_states=tuple(
                self.__capture_linear_state(
                    module_name,
                    linear_layer,
                    optimizer_parameter_ids,
                )
                for module_name, linear_layer in linear_modules.items()
            ),
        )

    def __linear_modules_for_step(
        self,
        target_step: int,
    ) -> dict[str, LinearAbstract]:
        linear_modules = dict(self._linear_modules)
        for (step, _module_key), (
            module_name,
            linear_layer,
        ) in self._activation_modules.items():
            if step != target_step:
                continue
            for tracked_name, tracked_layer in tuple(linear_modules.items()):
                if tracked_name == module_name or tracked_layer is linear_layer:
                    linear_modules.pop(tracked_name)
            linear_modules[module_name] = linear_layer
        return linear_modules

    @staticmethod
    def __optimizer_parameter_ids(optimizer: Optimizer) -> set[int] | None:
        if optimizer is None:
            return None
        return {
            id(parameter)
            for parameter_group in optimizer.param_groups
            for parameter in parameter_group["params"]
        }

    def __capture_linear_state(
        self,
        module_name: str,
        linear_layer: LinearAbstract,
        optimizer_parameter_ids: set[int] | None,
    ) -> _LinearBeforeStep:
        return _LinearBeforeStep(
            module_name=module_name,
            linear_layer=linear_layer,
            weights=self.__capture_parameter(
                linear_layer.weight_params,
                optimizer_parameter_ids,
            ),
            bias=(
                self.__capture_parameter(
                    linear_layer.bias_params,
                    optimizer_parameter_ids,
                )
                if linear_layer.bias_params is not None
                else None
            ),
        )

    @staticmethod
    def __capture_parameter(
        parameter: Parameter,
        optimizer_parameter_ids: set[int] | None,
    ) -> _ParameterBeforeStep:
        capture_gradient = optimizer_parameter_ids is None or (
            id(parameter) in optimizer_parameter_ids
        )
        local_gradient_summary = (
            _LinearDiagnostics.summarize(parameter.grad)
            if capture_gradient and parameter.grad is not None
            else None
        )
        return _ParameterBeforeStep(
            parameter=parameter,
            values=parameter.detach().clone(),
            gradient_summary=local_gradient_summary,
        )

    def __track_buffered_activations(
        self,
        pl_module: LightningModule,
        pending_step: _PendingOptimizerStep,
    ) -> None:
        for linear_state in pending_step.linear_states:
            input_summary = self.__pop_activation_summary(
                pending_step.step,
                linear_state.linear_layer,
                "input",
                linear_state.weights.values,
            )
            output_summary = self.__pop_activation_summary(
                pending_step.step,
                linear_state.linear_layer,
                "output",
                linear_state.weights.values,
            )
            self.__track_forward_diagnostics(
                pl_module,
                linear_state.module_name,
                input_summary,
                output_summary,
            )

    def __pop_activation_summary(
        self,
        step: int,
        linear_layer: LinearAbstract,
        channel: str,
        _reference: Tensor,
    ) -> _TensorSummary | None:
        moments = self._activation_moments.pop((step, id(linear_layer), channel), None)
        return moments.summarize() if moments is not None else None

    def __discard_activations_through(self, completed_step: int) -> None:
        for key in tuple(self._activation_moments):
            if key[0] <= completed_step:
                self._activation_moments.pop(key)
        for key in tuple(self._activation_modules):
            if key[0] <= completed_step:
                self._activation_modules.pop(key)

    def __build_tracking_contexts(
        self,
        pl_module: LightningModule,
        pending_step: _PendingOptimizerStep,
    ) -> tuple[_LinearTrackingContext, ...]:
        contexts = []
        for linear_state in pending_step.linear_states:
            linear_layer = linear_state.linear_layer
            weights = self.__build_parameter_channel_metrics(
                linear_layer.weight_params,
                linear_state.weights,
                include_update_ratio=True,
            )
            bias = None
            if linear_layer.bias_params is not None and linear_state.bias is not None:
                bias = self.__build_parameter_channel_metrics(
                    linear_layer.bias_params,
                    linear_state.bias,
                    include_update_ratio=False,
                )
            contexts.append(
                _LinearTrackingContext(
                    pl_module=pl_module,
                    module_name=linear_state.module_name,
                    weights=weights,
                    bias=bias,
                    input_feature_norms=_LinearDiagnostics.stable_norm(
                        weights.values,
                        dim=1,
                    ),
                    output_feature_norms=_LinearDiagnostics.stable_norm(
                        weights.values,
                        dim=0,
                    ),
                    weight_conditioning=(
                        _LinearDiagnostics.weight_conditioning(weights.values)
                        if self.log_weight_conditioning
                        else None
                    ),
                )
            )
        return tuple(contexts)

    def __build_parameter_channel_metrics(
        self,
        parameter: Parameter,
        before_step: _ParameterBeforeStep,
        *,
        include_update_ratio: bool,
    ) -> _LinearParameterChannelMetrics:
        current_values = parameter.detach()
        same_parameter = before_step.parameter is parameter
        change = None
        if same_parameter and before_step.values.shape == current_values.shape:
            delta_norm = _LinearDiagnostics.stable_norm(
                current_values - before_step.values
            )
            change = _ParameterChangeMetrics(
                delta_norm=delta_norm,
                relative_delta_norm=_LinearDiagnostics.safe_ratio(
                    delta_norm,
                    _LinearDiagnostics.stable_norm(before_step.values),
                ),
            )
        gradient_summary = before_step.gradient_summary if same_parameter else None
        return _LinearParameterChannelMetrics(
            values=current_values,
            summary=_LinearDiagnostics.summarize(current_values),
            change=change,
            gradient_summary=gradient_summary,
            update_ratio=(
                change.relative_delta_norm
                if include_update_ratio and change is not None
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
        self.__remove_discovery_hook()
        for hook_handle in self._hooks.values():
            hook_handle.remove()
        self._hooks.clear()
        self._linear_modules.clear()
        self._linear_module_names.clear()
        self._activation_moments.clear()
        self._activation_modules.clear()
        self._pending_step = None

    def __remove_discovery_hook(self) -> None:
        if self._discovery_hook is not None:
            self._discovery_hook.remove()
            self._discovery_hook = None

    def __install_discovery_hook(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if self._discovery_hook is None:
            self._discovery_hook = register_module_forward_pre_hook(
                self.__make_linear_discovery_hook(trainer, pl_module),
            )
