from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.callbacks import Callback

from emperor.layers._monitoring.callbacks._hooks import (
    _extract_hidden_tensor,
    _install_method_replacement,
    _MethodReplacement,
    _remove_hooks,
    _restore_method_replacements,
)
from emperor.layers._monitoring.diagnostics import (
    _RecurrentDiagnostics,
    _RecurrentObservation,
    _RecurrentTrackingContext,
)
from emperor.monitoring import MonitorEmissionPolicy, MonitorTensorHistory

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module
    from torch.utils.hooks import RemovableHandle


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
