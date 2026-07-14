from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from emperor.attention.core.runtime import QKV
from emperor.experiments.monitor_policy import MonitorEmissionPolicy

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor
    from torch.nn import Module


class AttentionMonitorCallback(Callback):
    DEAD_HEAD_ENTROPY_FLOOR = 1e-6

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_head_scalars: bool = False,
    ):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        if history_size <= 0:
            raise ValueError("history_size must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_head_scalars = log_per_head_scalars
        self._hooks = []
        self._wrapped_methods = []
        self._attention_modules = []
        self._traces: dict[int, dict[str, Any]] = {}
        self._entropy_history = {}
        self._max_probability_history = {}
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.attention.core.layers import MultiHeadAttentionAbstract

        self.__cleanup()
        self._emission_policy.clear()
        self._attention_modules.clear()
        for name, module in pl_module.named_modules():
            if not isinstance(module, MultiHeadAttentionAbstract):
                continue
            self._attention_modules.append((name, module))
            self._entropy_history[name] = []
            self._max_probability_history[name] = []
            self._hooks.append(
                module.register_forward_pre_hook(
                    self.__make_forward_pre_hook(name, module)
                )
            )
            self._hooks.append(
                module.register_forward_hook(
                    self.__make_forward_hook(name, module, pl_module)
                )
            )
            self.__wrap_projector(module)
            self.__wrap_processor(module, pl_module)

    def __make_forward_pre_hook(self, name: str, module: Module):
        def hook(layer: Module, inputs: tuple) -> None:
            self._traces[id(module)] = {"name": name}

        return hook

    def __wrap_projector(self, module: Module) -> None:
        projector = getattr(module, "projector", None)
        if projector is None or not hasattr(projector, "compute_qkv_projections"):
            return
        original = projector.compute_qkv_projections

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            trace = self._traces.setdefault(id(module), {})
            if isinstance(output, QKV):
                trace["qkv"] = (
                    output.query.detach(),
                    output.key.detach(),
                    output.value.detach(),
                )
            return output

        self.__replace_method(projector, "compute_qkv_projections", original, wrapped)

    def __wrap_processor(
        self,
        module: Module,
        pl_module: LightningModule,
    ) -> None:
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "compute_attention"):
            return

        original_compute_attention = processor.compute_attention

        def wrapped_compute_attention(*args, **kwargs):
            output = original_compute_attention(*args, **kwargs)
            trace = self._traces.setdefault(id(module), {})
            qkv = args[0] if len(args) > 0 else kwargs.get("qkv")
            attention_mask = (
                args[1]
                if len(args) > 1
                else kwargs.get("merged_attention_mask")
            )
            query = qkv.query if isinstance(qkv, QKV) else None
            key = qkv.key if isinstance(qkv, QKV) else None
            trace["attention_inputs"] = (
                query.detach() if torch.is_tensor(query) else None,
                key.detach() if torch.is_tensor(key) else None,
                attention_mask.detach() if torch.is_tensor(attention_mask) else None,
            )
            if self.__should_sample(pl_module):
                approximate = self.__compute_approximate_weights(
                    query,
                    key,
                    attention_mask,
                )
                if approximate is not None:
                    trace["approximate_weights"] = approximate
            return output

        self.__replace_method(
            processor,
            "compute_attention",
            original_compute_attention,
            wrapped_compute_attention,
        )

        for method_name in (
            "_SelfAttentionProcessor__compute_masked_attention_weights",
            "_MixtureOfAttentionHeadsProcessor__compute_masked_attention_weights",
        ):
            if not hasattr(processor, method_name):
                continue
            original = getattr(processor, method_name)

            def make_wrapped(original_method):
                def wrapped(*args, **kwargs):
                    weights = original_method(*args, **kwargs)
                    if torch.is_tensor(weights):
                        trace = self._traces.setdefault(id(module), {})
                        trace["exact_weights"] = weights.detach()
                    return weights

                return wrapped

            self.__replace_method(
                processor,
                method_name,
                original,
                make_wrapped(original),
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

    def __should_sample(self, module: LightningModule) -> bool:
        step = getattr(module, "global_step", 0)
        return step % self.log_every_n_steps == 0

    def __compute_approximate_weights(
        self,
        query: object,
        key: object,
        attention_mask: object,
    ) -> Tensor | None:
        if not torch.is_tensor(query) or not torch.is_tensor(key):
            return None
        if query.dim() not in (3, 4) or key.dim() not in (3, 4):
            return None
        query_values = query.detach().float()
        key_values = key.detach().float()
        raw_weights = torch.matmul(
            query_values * query_values.size(-1) ** -0.5,
            key_values.transpose(-2, -1),
        )
        if torch.is_tensor(attention_mask):
            mask = attention_mask.detach()
            try:
                if mask.dtype == torch.bool:
                    raw_weights = raw_weights.masked_fill(mask, -torch.inf)
                else:
                    raw_weights = raw_weights + mask.float()
            except RuntimeError:
                return None
        return F.softmax(raw_weights, dim=-1)

    def __make_forward_hook(
        self,
        name: str,
        module: Module,
        pl_module: LightningModule,
    ):
        def hook(layer: Module, inputs: tuple, output: object) -> None:
            if not self.__should_sample(pl_module):
                return
            trace = self._traces.setdefault(id(module), {"name": name})
            output_tensor, returned_weights, auxiliary_loss = (
                self.__parse_forward_output(output)
            )
            if output_tensor is not None:
                trace["output"] = output_tensor
            if returned_weights is not None and "exact_weights" not in trace:
                trace["exact_weights"] = returned_weights
            if auxiliary_loss is not None:
                trace["auxiliary_loss"] = auxiliary_loss
            self.__log_trace(pl_module, name, module, trace)

        return hook

    def __parse_forward_output(
        self,
        output: object,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if not isinstance(output, tuple):
            return (
                output.detach() if torch.is_tensor(output) else None,
                None,
                None,
            )
        output_tensor = output[0] if len(output) > 0 else None
        weights = output[1] if len(output) > 1 else None
        auxiliary_loss = output[2] if len(output) > 2 else None
        return (
            output_tensor.detach() if torch.is_tensor(output_tensor) else None,
            weights.detach() if torch.is_tensor(weights) else None,
            auxiliary_loss.detach() if torch.is_tensor(auxiliary_loss) else None,
        )

    def __log_trace(
        self,
        pl_module: LightningModule,
        name: str,
        module: Module,
        trace: dict[str, Any],
    ) -> None:
        prefix = f"{name}/attention"
        qkv = trace.get("qkv")
        if isinstance(qkv, tuple) and len(qkv) == 3:
            self.__log_tensor_norm_mean(pl_module, prefix, "q", qkv[0])
            self.__log_tensor_norm_mean(pl_module, prefix, "k", qkv[1])
            self.__log_tensor_norm_mean(pl_module, prefix, "v", qkv[2])

        output = trace.get("output")
        if torch.is_tensor(output):
            pl_module.log(f"{prefix}/output_norm", output.float().norm())

        auxiliary_loss = trace.get("auxiliary_loss")
        if torch.is_tensor(auxiliary_loss):
            pl_module.log(f"{prefix}/auxiliary_loss", auxiliary_loss.float().mean())

        pl_module.log(
            f"{prefix}/configured_dropout_probability",
            torch.tensor(float(getattr(module, "dropout_probability", 0.0))),
        )

        attention_inputs = trace.get("attention_inputs")
        mask = attention_inputs[2] if isinstance(attention_inputs, tuple) else None
        pl_module.log(f"{prefix}/mask_coverage", self.__mask_coverage(mask))

        exact_weights = trace.get("exact_weights")
        if torch.is_tensor(exact_weights):
            self.__log_weight_stats(
                pl_module,
                name,
                module,
                exact_weights,
                approximate=False,
            )
            pl_module.log(
                f"{prefix}/dropout_zero_fraction",
                (exact_weights.float() == 0.0).float().mean(),
            )
            return

        approximate_weights = trace.get("approximate_weights")
        if torch.is_tensor(approximate_weights):
            self.__log_weight_stats(
                pl_module,
                name,
                module,
                approximate_weights,
                approximate=True,
            )

    def __log_tensor_norm_mean(
        self,
        module: LightningModule,
        prefix: str,
        label: str,
        tensor: Tensor,
    ) -> None:
        values = tensor.detach().float()
        module.log(f"{prefix}/{label}_norm_mean", values.norm(dim=-1).mean())

    def __mask_coverage(self, mask: object) -> Tensor:
        if not torch.is_tensor(mask) or mask.numel() == 0:
            return torch.zeros(())
        if mask.dtype == torch.bool:
            return mask.detach().float().mean()
        return (mask.detach().float() != 0.0).float().mean()

    def __log_weight_stats(
        self,
        pl_module: LightningModule,
        name: str,
        module: Module,
        weights: Tensor,
        *,
        approximate: bool,
    ) -> None:
        if type(approximate) is not bool:
            raise TypeError("approximate must be a bool.")
        prefix = f"{name}/attention"
        metric_prefix = "approximate_" if approximate else ""
        per_head_entropy, per_head_max_probability = self.__per_head_stats(
            module,
            weights.detach().float(),
        )
        if per_head_entropy.numel() == 0:
            return

        pl_module.log(
            f"{prefix}/{metric_prefix}entropy_mean",
            per_head_entropy.mean(),
        )
        pl_module.log(
            f"{prefix}/{metric_prefix}max_probability_mean",
            per_head_max_probability.mean(),
        )
        pl_module.log(
            f"{prefix}/{metric_prefix}dead_head_fraction",
            (per_head_entropy <= self.DEAD_HEAD_ENTROPY_FLOOR).float().mean(),
        )

        if self.log_per_head_scalars:
            for head_idx, entropy in enumerate(per_head_entropy):
                pl_module.log(
                    f"{prefix}/head_{head_idx}/{metric_prefix}entropy",
                    entropy,
                )
            for head_idx, max_probability in enumerate(per_head_max_probability):
                pl_module.log(
                    f"{prefix}/head_{head_idx}/{metric_prefix}max_probability",
                    max_probability,
                )

        if approximate:
            return

        self.__append_history(self._entropy_history[name], per_head_entropy)
        self.__append_history(
            self._max_probability_history[name],
            per_head_max_probability,
        )
        self.__log_visual_summaries(
            pl_module,
            name,
            per_head_entropy,
            per_head_max_probability,
        )

    def __per_head_stats(
        self,
        module: Module,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        weights = self.__reshape_weights_by_head(module, weights)
        if weights is None or weights.numel() == 0:
            empty = torch.empty(0)
            return empty, empty
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(normalized.clamp_min(1e-12).log() * normalized).sum(dim=-1)
        max_probability = normalized.max(dim=-1).values
        reduce_dims = tuple(index for index in range(entropy.dim()) if index != 1)
        return entropy.mean(dim=reduce_dims), max_probability.mean(dim=reduce_dims)

    def __reshape_weights_by_head(
        self,
        module: Module,
        weights: Tensor,
    ) -> Tensor | None:
        num_heads = int(module.num_heads or 0)
        if num_heads <= 0:
            return None
        values = weights.detach().float()
        if values.dim() == 4:
            if values.size(1) == num_heads:
                return values
            if values.size(0) == num_heads:
                return values.unsqueeze(0).transpose(1, 0)
        if values.dim() == 5 and values.size(2) == num_heads:
            batch, top_k, heads, target, source = values.shape
            return values.view(batch * top_k, heads, target, source)
        if values.dim() == 3 and values.size(0) % num_heads == 0:
            return values.view(-1, num_heads, values.size(-2), values.size(-1))
        return None

    def __append_history(self, history: list[Tensor], values: Tensor) -> None:
        history.append(values.detach().float().cpu())
        del history[: -self.history_size]

    def __log_visual_summaries(
        self,
        module: LightningModule,
        name: str,
        entropy_by_head: Tensor,
        max_probability_by_head: Tensor,
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return
        step = getattr(module, "global_step", 0)
        self.__log_histogram(
            experiment,
            f"{name}/attention/histogram/entropy_by_head",
            entropy_by_head,
            step,
        )
        self.__log_histogram(
            experiment,
            f"{name}/attention/histogram/max_probability_by_head",
            max_probability_by_head,
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/attention/heatmap/entropy_by_head",
            self._entropy_history[name],
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/attention/heatmap/max_probability_by_head",
            self._max_probability_history[name],
            step,
        )

    def __log_histogram(
        self, experiment, tag: str, values: Tensor, step: int
    ) -> None:
        self._emission_policy.emit_histogram(experiment, tag, values, step)

    def __log_heatmap(
        self,
        experiment,
        tag: str,
        history: list[Tensor],
        step: int,
    ) -> None:
        if not hasattr(experiment, "add_image") or not history:
            return
        max_heads = max(vector.numel() for vector in history)
        if max_heads == 0:
            return
        padded = [F.pad(vector, (0, max_heads - vector.numel())) for vector in history]
        heatmap = torch.stack(padded).T
        heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        self._emission_policy.emit_image(
            experiment, tag, heatmap.unsqueeze(0), step, dataformats="CHW"
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__cleanup()

    def __cleanup(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for target, method_name, original in reversed(self._wrapped_methods):
            setattr(target, method_name, original)
        self._wrapped_methods.clear()
        self._attention_modules.clear()
        self._traces.clear()
        self._entropy_history.clear()
        self._max_probability_history.clear()
        self._emission_policy.clear()
