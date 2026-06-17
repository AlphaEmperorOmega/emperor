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


class ParametricLayerMonitorCallback(Callback):
    """Logs generated-parameter and mixture-router dynamics for ParametricLayer."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_slot_scalars: bool = False,
    ):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_slot_scalars = log_per_slot_scalars
        self._wrapped_methods = []
        self._parametric_modules = []
        self._traces: dict[int, dict[str, Any]] = {}
        self._weight_utilization_history = {}
        self._bias_utilization_history = {}
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.parametric.core.layers import ParametricLayer

        self._emission_policy.clear()
        self._parametric_modules.clear()
        for name, module in pl_module.named_modules():
            if not isinstance(module, ParametricLayer):
                continue
            self._parametric_modules.append((name, module))
            self._weight_utilization_history[name] = []
            self._bias_utilization_history[name] = []
            self.__wrap_forward(name, module, pl_module)
            self.__wrap_generate_parameters(module)
            self.__wrap_affine_callback(module)
            self.__wrap_sampling_methods(module)

    def __wrap_forward(
        self,
        name: str,
        module: "Module",
        pl_module: "LightningModule",
    ) -> None:
        original = module.forward

        def wrapped(*args, **kwargs):
            trace = {"name": name}
            self._traces[id(module)] = trace
            output = original(*args, **kwargs)
            if self.__should_sample(pl_module):
                self.__log_trace(pl_module, name, module, trace)
            return output

        self.__replace_method(module, "forward", original, wrapped)

    def __wrap_generate_parameters(self, module: "Module") -> None:
        original = module._generate_parameters

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            if isinstance(output, tuple) and len(output) == 4:
                weights, bias, skip_mask, loss = output
                trace = self._traces.setdefault(id(module), {})
                if torch.is_tensor(weights):
                    trace["weights"] = weights.detach()
                if torch.is_tensor(bias):
                    trace["bias"] = bias.detach()
                if torch.is_tensor(skip_mask):
                    trace["skip_mask"] = skip_mask.detach()
                if torch.is_tensor(loss):
                    trace["loss"] = loss.detach()
            return output

        self.__replace_method(module, "_generate_parameters", original, wrapped)

    def __wrap_affine_callback(self, module: "Module") -> None:
        original = module._compute_affine_transformation_callback

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            weights = args[0] if len(args) > 0 else kwargs.get("weights")
            bias = args[1] if len(args) > 1 else kwargs.get("bias")
            input_tensor = args[2] if len(args) > 2 else kwargs.get("input")
            trace = self._traces.setdefault(id(module), {})
            if torch.is_tensor(output):
                trace["affine_output"] = output.detach()
            if torch.is_tensor(input_tensor):
                trace["affine_input"] = input_tensor.detach()
            if torch.is_tensor(weights):
                trace["affine_weights"] = weights.detach()
            if torch.is_tensor(bias):
                trace["affine_bias"] = bias.detach()
            return output

        self.__replace_method(
            module,
            "_compute_affine_transformation_callback",
            original,
            wrapped,
        )

    def __wrap_sampling_methods(self, module: "Module") -> None:
        for method_name, slot in (
            ("_ParametricLayer__sample_weight_probabilities_and_indices", "weight"),
            ("_ParametricLayer__sample_bias_probabilities_and_indices", "bias"),
        ):
            if not hasattr(module, method_name):
                continue
            original = getattr(module, method_name)

            def make_wrapped(original_method, sample_slot: str):
                def wrapped(*args, **kwargs):
                    output = original_method(*args, **kwargs)
                    if isinstance(output, tuple) and len(output) == 4:
                        probabilities, indices, skip_mask, loss = output
                        trace = self._traces.setdefault(id(module), {})
                        trace[f"{sample_slot}_sample"] = {
                            "probabilities": (
                                probabilities.detach()
                                if torch.is_tensor(probabilities)
                                else None
                            ),
                            "indices": (
                                indices.detach() if torch.is_tensor(indices) else None
                            ),
                            "skip_mask": (
                                skip_mask.detach()
                                if torch.is_tensor(skip_mask)
                                else None
                            ),
                            "loss": loss.detach() if torch.is_tensor(loss) else None,
                        }
                    return output

                return wrapped

            self.__replace_method(
                module,
                method_name,
                original,
                make_wrapped(original, slot),
            )

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
        module: "Module",
        trace: dict[str, Any],
    ) -> None:
        prefix = f"{name}/parametric"
        weights = trace.get("weights")
        bias = trace.get("bias")
        if torch.is_tensor(weights):
            weight_values = weights.detach().float()
            pl_module.log(f"{prefix}/generated_weight_norm", weight_values.norm())
            pl_module.log(
                f"{prefix}/weight_clip_saturation_fraction",
                self.__clip_saturation(module.weight_mixture_model, weight_values),
            )
        if torch.is_tensor(bias):
            bias_values = bias.detach().float()
            pl_module.log(f"{prefix}/generated_bias_norm", bias_values.norm())
            pl_module.log(
                f"{prefix}/bias_clip_saturation_fraction",
                self.__clip_saturation(module.bias_mixture_model, bias_values),
            )

        loss = trace.get("loss")
        if torch.is_tensor(loss):
            pl_module.log(f"{prefix}/auxiliary_loss", loss.detach().float().mean())

        skip_mask = trace.get("skip_mask")
        if torch.is_tensor(skip_mask):
            retention = skip_mask.detach().float().mean()
            pl_module.log(f"{prefix}/skip_fraction", retention)
            pl_module.log(f"{prefix}/drop_fraction", 1.0 - retention)

        self.__log_affine_stats(pl_module, name, trace)
        self.__log_router_and_mixture_stats(pl_module, name, module, trace, "weight")
        self.__log_router_and_mixture_stats(pl_module, name, module, trace, "bias")

    def __clip_saturation(self, mixture: object, values: "Tensor") -> "Tensor":
        clip_range = getattr(mixture, "clip_range", None)
        if clip_range is None or clip_range <= 0:
            return values.new_zeros(())
        return (values.abs() >= float(clip_range)).float().mean()

    def __log_affine_stats(
        self,
        module: "LightningModule",
        name: str,
        trace: dict[str, Any],
    ) -> None:
        output = trace.get("affine_output")
        input_tensor = trace.get("affine_input")
        if not torch.is_tensor(output):
            return
        prefix = f"{name}/parametric/affine"
        output_values = output.detach().float()
        module.log(f"{prefix}/output_norm", output_values.norm())
        if torch.is_tensor(input_tensor):
            input_values = input_tensor.detach().float()
            module.log(
                f"{prefix}/relative_output_norm",
                output_values.norm() / input_values.norm().clamp_min(1e-6),
            )
            if input_values.shape == output_values.shape:
                delta = output_values - input_values
                module.log(f"{prefix}/delta_norm", delta.norm())
                module.log(
                    f"{prefix}/relative_delta_norm",
                    delta.norm() / input_values.norm().clamp_min(1e-6),
                )

    def __log_router_and_mixture_stats(
        self,
        pl_module: "LightningModule",
        name: str,
        module: "Module",
        trace: dict[str, Any],
        slot: str,
    ) -> None:
        sample = trace.get(f"{slot}_sample")
        if not isinstance(sample, dict):
            return
        probabilities = sample.get("probabilities")
        indices = sample.get("indices")
        loss = sample.get("loss")
        if torch.is_tensor(loss):
            pl_module.log(
                f"{name}/router/{slot}_auxiliary_loss",
                loss.detach().float().mean(),
            )
        if torch.is_tensor(probabilities):
            pl_module.log(
                f"{name}/router/{slot}_entropy",
                self.__router_entropy(probabilities.detach().float()),
            )
        utilization = self.__utilization(
            probabilities,
            indices,
            self.__num_experts(module, slot),
        )
        if utilization is None:
            return
        utilization = utilization.detach().float()
        active_slots = (utilization > 0).sum().float()
        prefix = f"{name}/mixture/{slot}"
        pl_module.log(f"{prefix}_active_slots", active_slots)
        pl_module.log(
            f"{prefix}_dead_slot_fraction",
            (utilization <= 0).float().mean(),
        )
        pl_module.log(f"{prefix}_max_utilization", utilization.max())
        pl_module.log(f"{prefix}_min_utilization", utilization.min())
        if self.log_per_slot_scalars:
            for slot_index, value in enumerate(utilization):
                pl_module.log(f"{prefix}_slot_{slot_index}_utilization", value)
        self.__append_utilization_history(name, slot, utilization)
        self.__log_visual_summaries(pl_module, name, slot, utilization)

    def __router_entropy(self, probabilities: "Tensor") -> "Tensor":
        values = probabilities.reshape(-1, probabilities.shape[-1]).float()
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(values.clamp_min(1e-12).log() * values).sum(dim=-1)
        return entropy.mean()

    def __num_experts(self, module: "Module", slot: str) -> int:
        mixture = (
            getattr(module, "weight_mixture_model", None)
            if slot == "weight"
            else getattr(module, "bias_mixture_model", None)
        )
        return int(getattr(mixture, "num_experts", 0) or 0)

    def __utilization(
        self,
        probabilities: object,
        indices: object,
        num_experts: int,
    ) -> "Tensor | None":
        if num_experts <= 0:
            return None
        if torch.is_tensor(indices):
            counts = torch.zeros(num_experts, device=indices.device)
            flat_indices = indices.detach().long().reshape(-1)
            valid = (flat_indices >= 0) & (flat_indices < num_experts)
            if valid.any():
                counts.scatter_add_(
                    0,
                    flat_indices[valid],
                    torch.ones_like(flat_indices[valid], dtype=counts.dtype),
                )
            return counts / counts.sum().clamp_min(1.0)
        if torch.is_tensor(probabilities) and probabilities.shape[-1] == num_experts:
            values = probabilities.detach().float().reshape(-1, num_experts)
            mass = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return mass.mean(dim=0)
        return None

    def __append_utilization_history(
        self,
        name: str,
        slot: str,
        utilization: "Tensor",
    ) -> None:
        history = (
            self._weight_utilization_history[name]
            if slot == "weight"
            else self._bias_utilization_history[name]
        )
        history.append(utilization.detach().float().cpu())
        del history[:-self.history_size]

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        slot: str,
        utilization: "Tensor",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return
        step = getattr(module, "global_step", 0)
        history = (
            self._weight_utilization_history[name]
            if slot == "weight"
            else self._bias_utilization_history[name]
        )
        self._emission_policy.emit_histogram(
            experiment,
            f"{name}/mixture/histogram/{slot}_utilization",
            utilization,
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/mixture/heatmap/{slot}_utilization",
            history,
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
        max_slots = max(vector.numel() for vector in history)
        if max_slots == 0:
            return
        padded = [
            F.pad(vector, (0, max_slots - vector.numel()))
            for vector in history
        ]
        heatmap = torch.stack(padded, dim=0).T
        heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        self._emission_policy.emit_image(
            experiment, tag, heatmap.unsqueeze(0), step, dataformats="CHW"
        )

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for target, method_name, original in reversed(self._wrapped_methods):
            setattr(target, method_name, original)
        self._wrapped_methods.clear()
        self._parametric_modules.clear()
        self._traces.clear()
        self._weight_utilization_history.clear()
        self._bias_utilization_history.clear()
        self._emission_policy.clear()
