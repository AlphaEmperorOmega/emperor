from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.layers._monitoring.diagnostics import (
    _LayerActivationTrackingContext,
    _LayerDropoutTrackingContext,
    _LayerGateTrackingContext,
    _LayerNormTrackingContext,
    _LayerResidualTrackingContext,
    _RecurrentDiagnostics,
    _RecurrentObservation,
    _RecurrentTrackingContext,
)
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
class _MethodReplacement:
    owner: object
    method_name: str
    original_method: Callable[..., object]

    def restore(self) -> None:
        setattr(self.owner, self.method_name, self.original_method)


def _install_method_replacement(
    replacements: list[_MethodReplacement],
    owner: object,
    method_name: str,
    original_method: Callable[..., object],
    replacement_method: Callable[..., object],
) -> None:
    setattr(owner, method_name, replacement_method)
    replacements.append(_MethodReplacement(owner, method_name, original_method))


def _restore_method_replacements(
    replacements: list[_MethodReplacement],
) -> None:
    for replacement in reversed(replacements):
        replacement.restore()
    replacements.clear()


def _remove_hooks(hook_handles: list[RemovableHandle]) -> None:
    for hook_handle in hook_handles:
        hook_handle.remove()
    hook_handles.clear()


def _extract_hidden_tensor(output: object) -> Tensor | None:
    if torch.is_tensor(output):
        return output
    hidden = getattr(output, "hidden", None)
    return hidden if torch.is_tensor(hidden) else None


class RecurrentLayerMonitorCallback(Callback):
    """Log recurrent-layer step dynamics without changing recurrent outputs."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_step_scalars: bool = False,
    ) -> None:
        super().__init__()
        self.__validate_positive("log_every_n_steps", log_every_n_steps)
        self.__validate_positive("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_step_scalars = log_per_step_scalars
        self._hooks: list[RemovableHandle] = []
        self._wrapped_methods: list[_MethodReplacement] = []
        self._observations: dict[int, _RecurrentObservation] = {}
        self._delta_history: dict[str, MonitorTensorHistory] = {}
        self._latest_gate_logits: dict[int, Tensor] = {}
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive(option_name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{option_name} must be greater than 0.")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.layers._recurrent import RecurrentLayer

        self.__cleanup()
        for module_name, recurrent_layer in pl_module.named_modules():
            if not isinstance(recurrent_layer, RecurrentLayer):
                continue
            self._delta_history[module_name] = MonitorTensorHistory(self.history_size)
            self.__attach_recurrent_gate_hook(recurrent_layer, pl_module)
            self.__wrap_forward(module_name, recurrent_layer, pl_module)
            self.__wrap_recurrent_controllers(recurrent_layer)

    def __attach_recurrent_gate_hook(
        self,
        recurrent_layer: Module,
        pl_module: LightningModule,
    ) -> None:
        recurrent_gate = getattr(recurrent_layer, "recurrent_gate", None)
        gate_model = getattr(recurrent_gate, "model", None)
        if gate_model is None:
            return
        self._hooks.append(
            gate_model.register_forward_hook(
                self.__make_gate_hook(recurrent_layer, pl_module)
            )
        )

    def __wrap_forward(
        self,
        module_name: str,
        recurrent_layer: Module,
        pl_module: LightningModule,
    ) -> None:
        original_forward = recurrent_layer.forward

        def monitored_forward(*args: object, **kwargs: object) -> object:
            layer_id = id(recurrent_layer)
            should_sample = self.__should_sample(pl_module)
            if should_sample:
                observation = _RecurrentObservation()
                self._observations[layer_id] = observation
            else:
                observation = None
                self._observations.pop(layer_id, None)
            self._latest_gate_logits.pop(layer_id, None)
            output = original_forward(*args, **kwargs)
            if observation is not None:
                self.__emit_observation(
                    pl_module,
                    module_name,
                    recurrent_layer,
                    observation,
                )
            return output

        _install_method_replacement(
            self._wrapped_methods,
            recurrent_layer,
            "forward",
            original_forward,
            monitored_forward,
        )

    def __wrap_recurrent_controllers(self, recurrent_layer: Module) -> None:
        method_name = "_RecurrentLayer__run_controllers"
        original_run_controllers = getattr(recurrent_layer, method_name)

        def monitored_run_controllers(
            *args: object,
            **kwargs: object,
        ) -> object:
            previous_hidden = (
                args[1] if len(args) > 1 else kwargs.get("previous_hidden")
            )
            output = original_run_controllers(*args, **kwargs)
            observation = self._observations.get(id(recurrent_layer))
            output_hidden = getattr(output, "hidden", None)
            if (
                observation is not None
                and torch.is_tensor(previous_hidden)
                and torch.is_tensor(output_hidden)
            ):
                self.__record_recurrent_step(
                    recurrent_layer,
                    observation,
                    previous_hidden,
                    output_hidden,
                )
            return output

        _install_method_replacement(
            self._wrapped_methods,
            recurrent_layer,
            method_name,
            original_run_controllers,
            monitored_run_controllers,
        )

    def __record_recurrent_step(
        self,
        recurrent_layer: Module,
        observation: _RecurrentObservation,
        previous_hidden: Tensor,
        output_hidden: Tensor,
    ) -> None:
        hidden_delta = output_hidden.detach().float() - previous_hidden.detach().float()
        observation.step_deltas.append(
            hidden_delta.reshape(hidden_delta.shape[0], -1).norm(dim=-1)
        )
        gate_logits = self._latest_gate_logits.pop(id(recurrent_layer), None)
        if gate_logits is None:
            return
        effective_gate_values = self.__effective_recurrent_gate_values(
            recurrent_layer,
            gate_logits.detach().float(),
        )
        observation.gate_values.append(effective_gate_values.reshape(-1))

    @staticmethod
    def __effective_recurrent_gate_values(
        recurrent_layer: Module,
        gate_logits: Tensor,
    ) -> Tensor:
        recurrent_gate = getattr(recurrent_layer, "recurrent_gate", None)
        if recurrent_gate is None or not hasattr(recurrent_gate, "effective_values"):
            return torch.sigmoid(gate_logits)
        return recurrent_gate.effective_values(gate_logits)

    def __make_gate_hook(
        self,
        recurrent_layer: Module,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def capture_gate_logits(
            _gate_model: Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            gate_logits = _extract_hidden_tensor(output)
            if gate_logits is not None:
                self._latest_gate_logits[id(recurrent_layer)] = gate_logits.detach()

        return capture_gate_logits

    def __should_sample(self, pl_module: LightningModule) -> bool:
        global_step = getattr(pl_module, "global_step", 0)
        return global_step % self.log_every_n_steps == 0

    def __emit_observation(
        self,
        pl_module: LightningModule,
        module_name: str,
        recurrent_layer: Module,
        observation: _RecurrentObservation,
    ) -> None:
        context = _RecurrentTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            metric_prefix=f"{module_name}/recurrent",
            recurrent_layer=recurrent_layer,
            metrics=_RecurrentDiagnostics.calculate(observation),
            device=getattr(pl_module, "device", torch.device("cpu")),
            experiment=getattr(
                getattr(pl_module, "logger", None),
                "experiment",
                None,
            ),
            global_step=getattr(pl_module, "global_step", 0),
        )
        self.__track_recurrent_diagnostics(context)

    def __track_recurrent_diagnostics(
        self,
        context: _RecurrentTrackingContext,
    ) -> None:
        self.__track_actual_steps(context)
        self.__track_hidden_delta_mean(context)
        self.__track_maximum_hidden_delta(context)
        self.__track_final_hidden_delta(context)
        self.__track_convergence_ratio(context)
        self.__track_maximum_step_fraction(context)
        self.__track_per_step_hidden_delta_mean(context)
        self.__track_gate_open_mean(context)
        self.__track_gate_open_fraction(context)
        self.__track_gate_saturation_fraction(context)
        self.__track_hidden_delta_history(context)
        self.__track_hidden_delta_histogram(context)
        self.__track_hidden_delta_heatmap(context)

    @staticmethod
    def __track_actual_steps(context: _RecurrentTrackingContext) -> None:
        actual_steps = (
            context.metrics.actual_steps if context.metrics is not None else 0
        )
        context.pl_module.log(
            f"{context.metric_prefix}/actual_steps",
            torch.tensor(float(actual_steps), device=context.device),
        )

    @staticmethod
    def __track_hidden_delta_mean(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/hidden_delta_mean",
            context.metrics.step_delta_means.mean(),
        )

    @staticmethod
    def __track_maximum_hidden_delta(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/hidden_delta_max",
            context.metrics.maximum_step_delta,
        )

    @staticmethod
    def __track_final_hidden_delta(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/hidden_delta_final",
            context.metrics.step_delta_means[-1],
        )

    @staticmethod
    def __track_convergence_ratio(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None:
            return
        step_delta_means = context.metrics.step_delta_means
        context.pl_module.log(
            f"{context.metric_prefix}/convergence_ratio",
            step_delta_means[-1] / step_delta_means[0].clamp_min(1e-12),
        )

    @staticmethod
    def __track_maximum_step_fraction(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None:
            return
        maximum_steps = max(
            float(getattr(context.recurrent_layer, "max_steps", 1)),
            1.0,
        )
        context.pl_module.log(
            f"{context.metric_prefix}/max_step_fraction",
            torch.tensor(
                context.metrics.actual_steps / maximum_steps,
                device=context.device,
            ),
        )

    def __track_per_step_hidden_delta_mean(
        self,
        context: _RecurrentTrackingContext,
    ) -> None:
        if not self.log_per_step_scalars or context.metrics is None:
            return
        for step_index, mean_delta in enumerate(context.metrics.step_delta_means):
            context.pl_module.log(
                f"{context.metric_prefix}/step_{step_index}/hidden_delta_mean",
                mean_delta,
            )

    @staticmethod
    def __track_gate_open_mean(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None or context.metrics.gate_values is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/gate/open_mean",
            context.metrics.gate_values.mean(),
        )

    @staticmethod
    def __track_gate_open_fraction(context: _RecurrentTrackingContext) -> None:
        if context.metrics is None or context.metrics.gate_values is None:
            return
        context.pl_module.log(
            f"{context.metric_prefix}/gate/open_fraction",
            (context.metrics.gate_values > 0.5).float().mean(),
        )

    @staticmethod
    def __track_gate_saturation_fraction(
        context: _RecurrentTrackingContext,
    ) -> None:
        if context.metrics is None or context.metrics.gate_values is None:
            return
        gate_values = context.metrics.gate_values
        context.pl_module.log(
            f"{context.metric_prefix}/gate/saturation_fraction",
            ((gate_values < 0.01) | (gate_values > 0.99)).float().mean(),
        )

    def __track_hidden_delta_history(
        self,
        context: _RecurrentTrackingContext,
    ) -> None:
        if context.metrics is None:
            return
        self._delta_history[context.module_name].append(
            context.metrics.step_delta_means
        )

    def __track_hidden_delta_histogram(
        self,
        context: _RecurrentTrackingContext,
    ) -> None:
        if context.metrics is None or context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.metric_prefix}/histogram/hidden_delta",
            context.metrics.flattened_step_deltas,
            context.global_step,
        )

    def __track_hidden_delta_heatmap(
        self,
        context: _RecurrentTrackingContext,
    ) -> None:
        if context.metrics is None or context.experiment is None:
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.metric_prefix}/heatmap/hidden_delta_by_step",
            self._delta_history[context.module_name],
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
        _remove_hooks(self._hooks)
        _restore_method_replacements(self._wrapped_methods)
        self._observations.clear()
        self._delta_history.clear()
        self._latest_gate_logits.clear()
        self._emission_policy.clear()


class LayerControllerMonitorCallback(Callback):
    """Log activation, gate, dropout, normalization, and residual diagnostics."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks: list[RemovableHandle] = []
        self._wrapped_methods: list[_MethodReplacement] = []
        self._hooked_gate_model_ids: set[int] = set()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.layers._layer import Layer

        self.__cleanup()
        for module_name, layer in pl_module.named_modules():
            if not isinstance(layer, Layer):
                continue
            self.__attach_gate_hook(module_name, layer, pl_module)
            self.__attach_dropout_hook(module_name, layer, pl_module)
            self.__attach_layer_norm_hook(module_name, layer, pl_module)
            if self.__should_track_activation(layer):
                self.__wrap_activation(module_name, layer, pl_module)
            self.__wrap_residual(module_name, layer, pl_module)

    @staticmethod
    def __should_track_activation(layer: Module) -> bool:
        from emperor.layers._options import ActivationOptions

        return layer.activation_function != ActivationOptions.DISABLED

    def __attach_gate_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        gate = getattr(layer, "gate_model", None)
        gate_model = getattr(gate, "model", None)
        if gate_model is None or id(gate_model) in self._hooked_gate_model_ids:
            return
        self._hooked_gate_model_ids.add(id(gate_model))
        self._hooks.append(
            gate_model.register_forward_hook(
                self.__make_gate_hook(module_name, layer, pl_module)
            )
        )

    def __attach_dropout_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        dropout_module = getattr(layer, "dropout_module", None)
        if dropout_module is not None:
            self._hooks.append(
                dropout_module.register_forward_hook(
                    self.__make_dropout_hook(module_name, pl_module)
                )
            )

    def __attach_layer_norm_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        layer_norm_module = getattr(layer, "layer_norm_module", None)
        if layer_norm_module is not None:
            self._hooks.append(
                layer_norm_module.register_forward_hook(
                    self.__make_layer_norm_hook(module_name, pl_module)
                )
            )

    def __make_gate_hook(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_gate_output(
            _gate_model: Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            raw_gate_values = _extract_hidden_tensor(output)
            if raw_gate_values is None:
                return
            detached_values = raw_gate_values.detach().float()
            effective_values = self.__effective_layer_gate_values(
                layer,
                detached_values,
            )
            context = _LayerGateTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                raw_values=detached_values,
                effective_values=effective_values,
            )
            self.__track_gate_diagnostics(context)

        return log_gate_output

    def __track_gate_diagnostics(
        self,
        context: _LayerGateTrackingContext,
    ) -> None:
        self.__track_raw_gate_mean(context)
        self.__track_raw_gate_variance(context)
        self.__track_raw_gate_positive_fraction(context)
        self.__track_raw_gate_saturation_fraction(context)
        self.__track_effective_gate_mean(context)
        self.__track_effective_gate_variance(context)
        self.__track_effective_gate_positive_fraction(context)
        self.__track_effective_gate_saturation_fraction(context)

    @staticmethod
    def __track_raw_gate_mean(context: _LayerGateTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/output_mean",
            context.raw_values.mean(),
        )

    @staticmethod
    def __track_raw_gate_variance(context: _LayerGateTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/output_var",
            context.raw_values.var(unbiased=False),
        )

    @staticmethod
    def __track_raw_gate_positive_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/gate/positive_fraction",
            (context.raw_values > 0).float().mean(),
        )

    @staticmethod
    def __track_raw_gate_saturation_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        raw_values = context.raw_values
        context.pl_module.log(
            f"{context.module_name}/gate/saturation_fraction",
            ((raw_values < -0.99) | (raw_values > 0.99)).float().mean(),
        )

    @staticmethod
    def __effective_layer_gate_values(
        layer: Module,
        raw_gate_values: Tensor,
    ) -> Tensor | None:
        gate = getattr(layer, "gate_model", None)
        if gate is None or not hasattr(gate, "effective_values"):
            return raw_gate_values
        return gate.effective_values(raw_gate_values)

    @staticmethod
    def __track_effective_gate_mean(context: _LayerGateTrackingContext) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_mean",
            context.effective_values.mean(),
        )

    @staticmethod
    def __track_effective_gate_variance(context: _LayerGateTrackingContext) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_var",
            context.effective_values.var(unbiased=False),
        )

    @staticmethod
    def __track_effective_gate_positive_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        if context.effective_values is None:
            return
        context.pl_module.log(
            f"{context.module_name}/gate/effective_positive_fraction",
            (context.effective_values > 0).float().mean(),
        )

    @staticmethod
    def __track_effective_gate_saturation_fraction(
        context: _LayerGateTrackingContext,
    ) -> None:
        if context.effective_values is None:
            return
        effective_values = context.effective_values
        saturation_mask = (effective_values < 0.01) | (effective_values > 0.99)
        context.pl_module.log(
            f"{context.module_name}/gate/effective_saturation_fraction",
            saturation_mask.float().mean(),
        )

    def __make_dropout_hook(
        self,
        module_name: str,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_dropout_output(
            _dropout_module: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                return
            context = _LayerDropoutTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                input_values=inputs[0].detach().float(),
                output_values=output.detach().float(),
            )
            self.__track_dropout_diagnostics(context)

        return log_dropout_output

    def __track_dropout_diagnostics(
        self,
        context: _LayerDropoutTrackingContext,
    ) -> None:
        self.__track_dropout_zero_fraction(context)
        self.__track_dropped_nonzero_fraction(context)

    @staticmethod
    def __track_dropout_zero_fraction(
        context: _LayerDropoutTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/dropout/zero_fraction",
            (context.output_values == 0.0).float().mean(),
        )

    @staticmethod
    def __track_dropped_nonzero_fraction(
        context: _LayerDropoutTrackingContext,
    ) -> None:
        nonzero_input = context.input_values != 0.0
        if nonzero_input.any():
            context.pl_module.log(
                f"{context.module_name}/dropout/dropped_nonzero_fraction",
                ((context.output_values == 0.0) & nonzero_input).float().sum()
                / nonzero_input.float().sum().clamp_min(1.0),
            )

    def __make_layer_norm_hook(
        self,
        module_name: str,
        pl_module: LightningModule,
    ) -> Callable[[Module, tuple[object, ...], object], None]:
        def log_layer_norm_output(
            _layer_norm: Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            if not self.__should_sample(pl_module):
                return
            if (
                not inputs
                or not torch.is_tensor(inputs[0])
                or not torch.is_tensor(output)
            ):
                return
            context = _LayerNormTrackingContext(
                pl_module=pl_module,
                module_name=module_name,
                input_values=inputs[0].detach().float(),
                output_values=output.detach().float(),
            )
            self.__track_layer_norm_diagnostics(context)

        return log_layer_norm_output

    def __track_layer_norm_diagnostics(
        self,
        context: _LayerNormTrackingContext,
    ) -> None:
        self.__track_layer_norm_output_mean(context)
        self.__track_layer_norm_output_variance(context)
        self.__track_layer_norm_relative_delta_norm(context)

    @staticmethod
    def __track_layer_norm_output_mean(context: _LayerNormTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/layer_norm/output_mean",
            context.output_values.mean(),
        )

    @staticmethod
    def __track_layer_norm_output_variance(
        context: _LayerNormTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/layer_norm/output_var",
            context.output_values.var(unbiased=False),
        )

    @staticmethod
    def __track_layer_norm_relative_delta_norm(
        context: _LayerNormTrackingContext,
    ) -> None:
        if context.input_values.shape != context.output_values.shape:
            return
        output_delta = context.output_values - context.input_values
        context.pl_module.log(
            f"{context.module_name}/layer_norm/relative_delta_norm",
            output_delta.norm() / context.input_values.norm().clamp_min(1e-6),
        )

    def __wrap_activation(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        method_name = "_Layer__maybe_apply_activation"
        original_activation = getattr(layer, method_name)

        def monitored_activation(*args: object, **kwargs: object) -> object:
            output = original_activation(*args, **kwargs)
            if self.__should_sample(pl_module) and torch.is_tensor(output):
                context = _LayerActivationTrackingContext(
                    pl_module=pl_module,
                    module_name=module_name,
                    activation_values=output.detach().float(),
                )
                self.__track_activation_diagnostics(context)
            return output

        _install_method_replacement(
            self._wrapped_methods,
            layer,
            method_name,
            original_activation,
            monitored_activation,
        )

    def __track_activation_diagnostics(
        self,
        context: _LayerActivationTrackingContext,
    ) -> None:
        self.__track_activation_zero_fraction(context)
        self.__track_activation_saturation_fraction(context)

    @staticmethod
    def __track_activation_zero_fraction(
        context: _LayerActivationTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/activation/zero_fraction",
            (context.activation_values == 0.0).float().mean(),
        )

    @staticmethod
    def __track_activation_saturation_fraction(
        context: _LayerActivationTrackingContext,
    ) -> None:
        activation_values = context.activation_values
        context.pl_module.log(
            f"{context.module_name}/activation/saturation_fraction",
            ((activation_values < -0.99) | (activation_values > 0.99)).float().mean(),
        )

    def __wrap_residual(
        self,
        module_name: str,
        layer: Module,
        pl_module: LightningModule,
    ) -> None:
        method_name = "_Layer__maybe_apply_residual_connection"
        if getattr(layer, "residual_connection", None) is None:
            return
        original_residual = getattr(layer, method_name)

        def monitored_residual(*args: object, **kwargs: object) -> object:
            output = original_residual(*args, **kwargs)
            input_values = args[0] if args else kwargs.get("input")
            previous_values = args[1] if len(args) > 1 else kwargs.get("prev_input")
            if self.__can_log_residual(
                pl_module,
                output,
                input_values,
                previous_values,
            ):
                context = _LayerResidualTrackingContext(
                    pl_module=pl_module,
                    module_name=module_name,
                    output_values=output.detach().float(),
                    input_values=input_values.detach().float(),
                    previous_values=previous_values.detach().float(),
                )
                self.__track_residual_diagnostics(context)
            return output

        _install_method_replacement(
            self._wrapped_methods,
            layer,
            method_name,
            original_residual,
            monitored_residual,
        )

    def __can_log_residual(
        self,
        pl_module: LightningModule,
        output: object,
        input_values: object,
        previous_values: object,
    ) -> bool:
        return (
            self.__should_sample(pl_module)
            and torch.is_tensor(output)
            and torch.is_tensor(input_values)
            and torch.is_tensor(previous_values)
            and output.shape == input_values.shape
        )

    def __track_residual_diagnostics(
        self,
        context: _LayerResidualTrackingContext,
    ) -> None:
        self.__track_residual_contribution_ratio(context)
        self.__track_residual_input_ratio(context)

    @staticmethod
    def __track_residual_contribution_ratio(
        context: _LayerResidualTrackingContext,
    ) -> None:
        residual_contribution = context.output_values - context.input_values
        context.pl_module.log(
            f"{context.module_name}/residual/contribution_ratio",
            residual_contribution.norm() / context.output_values.norm().clamp_min(1e-6),
        )

    @staticmethod
    def __track_residual_input_ratio(
        context: _LayerResidualTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/residual/input_ratio",
            context.previous_values.norm()
            / context.input_values.norm().clamp_min(1e-6),
        )

    def __should_sample(self, pl_module: LightningModule) -> bool:
        global_step = getattr(pl_module, "global_step", 0)
        return global_step % self.log_every_n_steps == 0

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
        _remove_hooks(self._hooks)
        _restore_method_replacements(self._wrapped_methods)
        self._hooked_gate_model_ids.clear()
