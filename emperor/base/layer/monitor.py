from __future__ import annotations

import torch
import torch.nn.functional as F

from lightning.pytorch.callbacks import Callback
from emperor.experiments.monitor_policy import MonitorEmissionPolicy

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from lightning import LightningModule, Trainer
    from torch.nn import Module


class RecurrentLayerMonitorCallback(Callback):
    """Logs recurrent-layer step dynamics without changing recurrent outputs."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_step_scalars: bool = False,
    ):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_step_scalars = log_per_step_scalars
        self._hooks = []
        self._wrapped_methods = []
        self._recurrent_modules = []
        self._traces: dict[int, dict[str, Any]] = {}
        self._delta_history = {}
        self._latest_gate_logits = {}
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.base.layer.recurrent import RecurrentLayer

        self._emission_policy.clear()
        self._recurrent_modules.clear()
        for name, module in pl_module.named_modules():
            if not isinstance(module, RecurrentLayer):
                continue
            self._recurrent_modules.append((name, module))
            self._delta_history[name] = []
            if self.__should_track_recurrent_gate(module):
                recurrent_gate = module.recurrent_gate
                self._hooks.append(
                    recurrent_gate.model.register_forward_hook(
                        self.__make_gate_hook(module)
                    )
                )
            self.__wrap_forward(name, module, pl_module)
            self.__wrap_recurrent_controllers(module)
            if self.__should_track_recurrent_halting(module):
                self.__wrap_preserve_methods(module)

    def __should_track_recurrent_gate(self, module: "Module") -> bool:
        recurrent_gate = getattr(module, "recurrent_gate", None)
        return getattr(recurrent_gate, "model", None) is not None

    def __should_track_recurrent_halting(self, module: "Module") -> bool:
        return getattr(module, "halting_model", None) is not None

    def __wrap_forward(
        self,
        name: str,
        module: "Module",
        pl_module: "LightningModule",
    ) -> None:
        original = module.forward

        def wrapped(*args, **kwargs):
            trace = {
                "name": name,
                "step_deltas": [],
                "gate_values": [],
                "preserved_fractions": [],
            }
            self._traces[id(module)] = trace
            output = original(*args, **kwargs)
            if self.__should_sample(pl_module):
                self.__log_trace(pl_module, name, module, trace)
            return output

        self.__replace_method(module, "forward", original, wrapped)

    def __wrap_recurrent_controllers(self, module: "Module") -> None:
        method_name = "_RecurrentLayer__run_recurrent_controllers"
        if not hasattr(module, method_name):
            return
        original = getattr(module, method_name)

        def wrapped(*args, **kwargs):
            candidate_state = args[0] if len(args) > 0 else kwargs.get("candidate_state")
            previous_hidden = args[1] if len(args) > 1 else kwargs.get("previous_hidden")
            output = original(*args, **kwargs)
            if torch.is_tensor(previous_hidden) and torch.is_tensor(
                getattr(output, "hidden", None)
            ):
                delta = (
                    output.hidden.detach().float()
                    - previous_hidden.detach().float()
                )
                step_delta = delta.reshape(delta.shape[0], -1).norm(dim=-1)
                trace = self._traces.setdefault(id(module), {})
                trace.setdefault("step_deltas", []).append(step_delta.detach())
                if self.log_per_step_scalars:
                    trace.setdefault("candidate_deltas", []).append(
                        self.__candidate_delta(candidate_state, previous_hidden)
                    )
                gate_logits = self._latest_gate_logits.pop(id(module), None)
                if torch.is_tensor(gate_logits):
                    gate_values = self.__effective_recurrent_gate_values(
                        module,
                        gate_logits.detach().float(),
                    )
                    if torch.is_tensor(gate_values):
                        trace.setdefault("gate_values", []).append(
                            gate_values.reshape(-1)
                        )
            return output

        self.__replace_method(module, method_name, original, wrapped)

    def __candidate_delta(
        self,
        candidate_state: object,
        previous_hidden: object,
    ) -> "Tensor | None":
        candidate = getattr(candidate_state, "hidden", None)
        if not torch.is_tensor(candidate) or not torch.is_tensor(previous_hidden):
            return None
        delta = candidate.detach().float() - previous_hidden.detach().float()
        return delta.reshape(delta.shape[0], -1).norm(dim=-1).mean()

    def __effective_recurrent_gate_values(
        self,
        module: "Module",
        gate_logits: "Tensor",
    ) -> "Tensor | None":
        recurrent_gate = getattr(module, "recurrent_gate", None)
        if recurrent_gate is None or not hasattr(recurrent_gate, "effective_values"):
            return torch.sigmoid(gate_logits)
        return recurrent_gate.effective_values(gate_logits)

    def __wrap_preserve_methods(self, module: "Module") -> None:
        for method_name in ("_RecurrentLayer__preserve_halted_hidden",):
            if not hasattr(module, method_name):
                continue
            original = getattr(module, method_name)

            def make_wrapped(original_method, name=method_name):
                def wrapped(*args, **kwargs):
                    halt_mask = args[2] if len(args) > 2 else kwargs.get("halt_mask")
                    output = original_method(*args, **kwargs)
                    if torch.is_tensor(halt_mask):
                        trace = self._traces.setdefault(id(module), {})
                        trace.setdefault("preserved_fractions", []).append(
                            halt_mask.detach().float().mean()
                        )
                    return output

                return wrapped

            self.__replace_method(module, method_name, original, make_wrapped(original))

    def __make_gate_hook(self, module: "Module"):
        def hook(submodule: "Module", inputs: tuple, output: object) -> None:
            hidden = self.__extract_hidden_tensor(output)
            if hidden is not None:
                self._latest_gate_logits[id(module)] = hidden.detach()

        return hook

    def __extract_hidden_tensor(self, output: object) -> "Tensor | None":
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        if torch.is_tensor(hidden):
            return hidden
        return None

    def __replace_method(
        self,
        target: object,
        method_name: str,
        original,
        wrapped,
    ) -> None:
        setattr(target, method_name, wrapped)
        self._wrapped_methods.append((target, method_name, original))

    def __should_sample(self, module: "LightningModule") -> bool:
        step = getattr(module, "global_step", 0)
        return step % self.log_every_n_steps == 0

    def __log_trace(
        self,
        pl_module: "LightningModule",
        name: str,
        recurrent_layer: "Module",
        trace: dict[str, Any],
    ) -> None:
        prefix = f"{name}/recurrent"
        step_deltas = [
            item.detach().float().reshape(-1)
            for item in trace.get("step_deltas", [])
            if torch.is_tensor(item)
        ]
        actual_steps = len(step_deltas)
        device = getattr(pl_module, "device", torch.device("cpu"))
        pl_module.log(
            f"{prefix}/actual_steps",
            torch.tensor(float(actual_steps), device=device),
        )
        if actual_steps == 0:
            return

        step_delta_means = torch.stack([delta.mean() for delta in step_deltas])
        step_delta_maxes = torch.stack([delta.max() for delta in step_deltas])
        pl_module.log(f"{prefix}/hidden_delta_mean", step_delta_means.mean())
        pl_module.log(f"{prefix}/hidden_delta_max", step_delta_maxes.max())
        pl_module.log(f"{prefix}/hidden_delta_final", step_delta_means[-1])
        pl_module.log(
            f"{prefix}/convergence_ratio",
            step_delta_means[-1] / step_delta_means[0].clamp_min(1e-12),
        )
        pl_module.log(
            f"{prefix}/max_step_fraction",
            torch.tensor(
                actual_steps / max(float(getattr(recurrent_layer, "max_steps", 1)), 1.0),
                device=device,
            ),
        )
        if self.log_per_step_scalars:
            for index, delta in enumerate(step_delta_means):
                pl_module.log(f"{prefix}/step_{index}/hidden_delta_mean", delta)

        gate_values = [
            item.detach().float().reshape(-1)
            for item in trace.get("gate_values", [])
            if torch.is_tensor(item)
        ]
        if gate_values:
            gates = torch.cat(gate_values)
            pl_module.log(f"{prefix}/gate/open_mean", gates.mean())
            pl_module.log(f"{prefix}/gate/open_fraction", (gates > 0.5).float().mean())
            pl_module.log(
                f"{prefix}/gate/saturation_fraction",
                ((gates < 0.01) | (gates > 0.99)).float().mean(),
            )

        preserved = [
            item.detach().float()
            for item in trace.get("preserved_fractions", [])
            if torch.is_tensor(item)
        ]
        if preserved:
            pl_module.log(
                f"{prefix}/preserved_halted_hidden_fraction",
                torch.stack(preserved).mean(),
            )

        history_vector = step_delta_means.detach().float().cpu()
        self.__append_history(self._delta_history[name], history_vector)
        self.__log_visual_summaries(pl_module, name, step_deltas, history_vector)

    def __append_history(self, history: list["Tensor"], values: "Tensor") -> None:
        history.append(values.detach().float().cpu())
        del history[:-self.history_size]

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        step_deltas: list["Tensor"],
        history_vector: "Tensor",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return
        step = getattr(module, "global_step", 0)
        self._emission_policy.emit_histogram(
            experiment,
            f"{name}/recurrent/histogram/hidden_delta",
            torch.cat([delta.cpu() for delta in step_deltas]),
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/recurrent/heatmap/hidden_delta_by_step",
            self._delta_history[name],
            step,
        )

    def __log_heatmap(
        self,
        experiment,
        tag: str,
        history: list["Tensor"],
        step: int,
    ) -> None:
        if not hasattr(experiment, "add_image") or not history:
            return
        max_steps = max(vector.numel() for vector in history)
        if max_steps == 0:
            return
        padded = [
            F.pad(vector, (0, max_steps - vector.numel()))
            for vector in history
        ]
        heatmap = torch.stack(padded, dim=0).T
        heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        self._emission_policy.emit_image(
            experiment, tag, heatmap.unsqueeze(0), step, dataformats="CHW"
        )

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for target, method_name, original in reversed(self._wrapped_methods):
            setattr(target, method_name, original)
        self._wrapped_methods.clear()
        self._recurrent_modules.clear()
        self._traces.clear()
        self._delta_history.clear()
        self._latest_gate_logits.clear()
        self._emission_policy.clear()


class LayerControllerMonitorCallback(Callback):
    """Logs controller pieces owned by base Layer modules."""

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self._hooks = []
        self._wrapped_methods = []
        self._layer_modules = []
        self._hooked_gate_model_ids = set()

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.base.layer.layer import Layer

        self._layer_modules.clear()
        self._hooked_gate_model_ids.clear()
        for name, module in pl_module.named_modules():
            if not isinstance(module, Layer):
                continue
            self._layer_modules.append((name, module))
            self.__attach_gate_hook(name, module, pl_module)
            self.__attach_dropout_hook(name, module, pl_module)
            self.__attach_layer_norm_hook(name, module, pl_module)
            if self.__should_track_activation(module):
                self.__wrap_activation(name, module, pl_module)
            self.__wrap_residual(name, module, pl_module)

    def __should_track_activation(self, layer: "Module") -> bool:
        from emperor.base.options import ActivationOptions

        return getattr(layer, "activation_function", None) != ActivationOptions.DISABLED

    def __attach_gate_hook(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ) -> None:
        gate_model = getattr(layer, "gate_model", None)
        gate_controller_model = getattr(gate_model, "model", None)
        if gate_controller_model is None or not self.__should_track_layer_gate(layer):
            return
        gate_model_id = id(gate_controller_model)
        if gate_model_id in self._hooked_gate_model_ids:
            return
        self._hooked_gate_model_ids.add(gate_model_id)
        self._hooks.append(
            gate_controller_model.register_forward_hook(
                self.__make_gate_hook(name, layer, pl_module)
            )
        )

    def __should_track_layer_gate(self, layer: "Module") -> bool:
        return getattr(layer, "gate_model", None) is not None

    def __attach_dropout_hook(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ) -> None:
        dropout_module = getattr(layer, "dropout_module", None)
        if dropout_module is None:
            return
        self._hooks.append(
            dropout_module.register_forward_hook(
                self.__make_dropout_hook(name, pl_module)
            )
        )

    def __attach_layer_norm_hook(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ) -> None:
        layer_norm_module = getattr(layer, "layer_norm_module", None)
        if layer_norm_module is None:
            return
        self._hooks.append(
            layer_norm_module.register_forward_hook(
                self.__make_layer_norm_hook(name, pl_module)
            )
        )

    def __make_gate_hook(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ):
        def hook(submodule: "Module", inputs: tuple, output: object) -> None:
            if not self.__should_sample(pl_module):
                return
            gate = self.__extract_hidden_tensor(output)
            if gate is None:
                return
            values = gate.detach().float()
            prefix = f"{name}/gate"
            pl_module.log(f"{prefix}/output_mean", values.mean())
            pl_module.log(f"{prefix}/output_var", values.var(unbiased=False))
            pl_module.log(f"{prefix}/positive_fraction", (values > 0).float().mean())
            pl_module.log(
                f"{prefix}/saturation_fraction",
                ((values < -0.99) | (values > 0.99)).float().mean(),
            )
            effective = self.__effective_layer_gate_values(layer, values)
            if effective is None:
                return
            pl_module.log(f"{prefix}/effective_mean", effective.mean())
            pl_module.log(
                f"{prefix}/effective_var",
                effective.var(unbiased=False),
            )
            pl_module.log(
                f"{prefix}/effective_positive_fraction",
                (effective > 0).float().mean(),
            )
            pl_module.log(
                f"{prefix}/effective_saturation_fraction",
                self.__effective_saturation_mask(layer, effective).float().mean(),
            )

        return hook

    def __effective_layer_gate_values(
        self,
        layer: "Module",
        values: "Tensor",
    ) -> "Tensor | None":
        gate = getattr(layer, "gate_model", None)
        if gate is None or not hasattr(gate, "effective_values"):
            return values
        return gate.effective_values(values)

    def __effective_saturation_mask(
        self,
        layer: "Module",
        values: "Tensor",
    ) -> "Tensor":
        return (values < 0.01) | (values > 0.99)

    def __make_dropout_hook(self, name: str, pl_module: "LightningModule"):
        def hook(submodule: "Module", inputs: tuple, output: object) -> None:
            if not self.__should_sample(pl_module):
                return
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            input_tensor = inputs[0].detach().float()
            output_tensor = output.detach().float()
            nonzero_input = input_tensor != 0.0
            prefix = f"{name}/dropout"
            pl_module.log(
                f"{prefix}/zero_fraction",
                (output_tensor == 0.0).float().mean(),
            )
            if nonzero_input.any():
                pl_module.log(
                    f"{prefix}/dropped_nonzero_fraction",
                    ((output_tensor == 0.0) & nonzero_input).float().sum()
                    / nonzero_input.float().sum().clamp_min(1.0),
                )

        return hook

    def __make_layer_norm_hook(self, name: str, pl_module: "LightningModule"):
        def hook(submodule: "Module", inputs: tuple, output: object) -> None:
            if not self.__should_sample(pl_module):
                return
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                return
            input_tensor = inputs[0].detach().float()
            output_tensor = output.detach().float()
            prefix = f"{name}/layer_norm"
            pl_module.log(f"{prefix}/output_mean", output_tensor.mean())
            pl_module.log(f"{prefix}/output_var", output_tensor.var(unbiased=False))
            if input_tensor.shape == output_tensor.shape:
                delta = output_tensor - input_tensor
                pl_module.log(
                    f"{prefix}/relative_delta_norm",
                    delta.norm() / input_tensor.norm().clamp_min(1e-6),
                )

        return hook

    def __wrap_activation(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ) -> None:
        method_name = "_Layer__maybe_apply_activation"
        if not hasattr(layer, method_name):
            return
        original = getattr(layer, method_name)

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            if self.__should_sample(pl_module) and torch.is_tensor(output):
                values = output.detach().float()
                prefix = f"{name}/activation"
                pl_module.log(
                    f"{prefix}/zero_fraction",
                    (values == 0.0).float().mean(),
                )
                pl_module.log(
                    f"{prefix}/saturation_fraction",
                    ((values < -0.99) | (values > 0.99)).float().mean(),
                )
            return output

        self.__replace_method(layer, method_name, original, wrapped)

    def __wrap_residual(
        self,
        name: str,
        layer: "Module",
        pl_module: "LightningModule",
    ) -> None:
        method_name = "_Layer__maybe_apply_residual_connection"
        if not hasattr(layer, method_name):
            return
        if getattr(layer, "residual_connection", None) is None:
            return
        original = getattr(layer, method_name)

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            input_tensor = args[0] if len(args) > 0 else kwargs.get("input")
            previous = args[1] if len(args) > 1 else kwargs.get("prev_input")
            if (
                self.__should_sample(pl_module)
                and torch.is_tensor(output)
                and torch.is_tensor(input_tensor)
                and torch.is_tensor(previous)
                and output.shape == input_tensor.shape
            ):
                contribution = output.detach().float() - input_tensor.detach().float()
                prefix = f"{name}/residual"
                pl_module.log(
                    f"{prefix}/contribution_ratio",
                    contribution.norm()
                    / output.detach().float().norm().clamp_min(1e-6),
                )
                pl_module.log(
                    f"{prefix}/input_ratio",
                    previous.detach().float().norm()
                    / input_tensor.detach().float().norm().clamp_min(1e-6),
                )
            return output

        self.__replace_method(layer, method_name, original, wrapped)

    def __extract_hidden_tensor(self, output: object) -> "Tensor | None":
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        if torch.is_tensor(hidden):
            return hidden
        return None

    def __replace_method(
        self,
        target: object,
        method_name: str,
        original,
        wrapped,
    ) -> None:
        setattr(target, method_name, wrapped)
        self._wrapped_methods.append((target, method_name, original))

    def __should_sample(self, module: "LightningModule") -> bool:
        step = getattr(module, "global_step", 0)
        return step % self.log_every_n_steps == 0

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for target, method_name, original in reversed(self._wrapped_methods):
            setattr(target, method_name, original)
        self._wrapped_methods.clear()
        self._layer_modules.clear()
        self._hooked_gate_model_ids.clear()
