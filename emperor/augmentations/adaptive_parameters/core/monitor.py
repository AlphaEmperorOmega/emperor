import torch
import torch.nn.functional as F

from lightning.pytorch.callbacks import Callback
from emperor.experiments.monitor_policy import MonitorEmissionPolicy
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch.nn import Module


class AdaptiveParameterMonitorCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_histograms: bool = False,
        log_internal_stats: bool = True,
    ):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.log_histograms = log_histograms
        self.log_internal_stats = log_internal_stats
        self._hooks = []
        self._adaptive_modules = []
        self._monitored_options = []
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.augmentations.adaptive_parameters.model import (
            AdaptiveParameterAugmentation,
        )

        self._emission_policy.clear()
        for name, module in pl_module.named_modules():
            if not isinstance(module, AdaptiveParameterAugmentation):
                continue
            self._adaptive_modules.append((name, module))
            for slot_name, metric_slot in (
                ("weight_model", "weight"),
                ("diagonal_model", "diagonal"),
                ("bias_model", "bias"),
                ("mask_model", "mask"),
            ):
                option = getattr(module, slot_name, None)
                if option is None:
                    continue
                hook = option.register_forward_hook(
                    self.__make_forward_hook(name, metric_slot, pl_module)
                )
                self._hooks.append(hook)
                self._monitored_options.append((name, metric_slot, option))

    def __make_forward_hook(
        self,
        augmentation_path: str,
        slot: str,
        lightning_module: "LightningModule",
    ):
        log_every_n_steps = self.log_every_n_steps

        def hook(option: "Module", inputs: tuple, output: Tensor) -> None:
            step = getattr(lightning_module, "global_step", 0)
            if step % log_every_n_steps != 0:
                return
            if not torch.is_tensor(output):
                return

            output_tensor = output.detach()
            prefix = f"{augmentation_path}/{slot}/batch"
            self.__log_common_stats(lightning_module, prefix, "output", output_tensor)

            base_tensor = self.__extract_base_tensor(inputs)
            delta_tensor = None
            if base_tensor is not None:
                base_tensor = base_tensor.detach()
                delta_tensor = output_tensor - base_tensor
                self.__log_base_and_delta_stats(
                    lightning_module, prefix, base_tensor, delta_tensor
                )

            self.__log_input_adaptivity_stats(
                lightning_module, prefix, output_tensor, delta_tensor
            )

            if self.log_internal_stats:
                self.__log_internal_stats(
                    lightning_module,
                    prefix,
                    slot,
                    option,
                    base_tensor,
                    output_tensor,
                )

            if self.log_histograms:
                self.__log_histograms(
                    lightning_module, prefix, output_tensor, delta_tensor
                )

        return hook

    def __extract_base_tensor(self, inputs: tuple) -> Tensor | None:
        if not inputs:
            return None
        base_tensor = inputs[0]
        if not torch.is_tensor(base_tensor):
            return None
        return base_tensor

    def __log_common_stats(
        self,
        module: "LightningModule",
        prefix: str,
        metric_prefix: str,
        tensor: Tensor,
    ) -> None:
        values = tensor.float()
        module.log(f"{prefix}/{metric_prefix}_mean", values.mean())
        module.log(f"{prefix}/{metric_prefix}_var", values.var(unbiased=False))
        module.log(f"{prefix}/{metric_prefix}_min", values.min())
        module.log(f"{prefix}/{metric_prefix}_max", values.max())
        module.log(f"{prefix}/{metric_prefix}_l2_norm", values.norm())
        module.log(f"{prefix}/{metric_prefix}_max_abs", values.abs().max())

    def __log_input_adaptivity_stats(
        self,
        module: "LightningModule",
        prefix: str,
        output_tensor: Tensor,
        delta_tensor: Tensor | None,
    ) -> None:
        adaptivity_tensor = (
            delta_tensor if delta_tensor is not None else output_tensor
        )
        if adaptivity_tensor.dim() == 0 or adaptivity_tensor.shape[0] < 2:
            return
        batch_size = adaptivity_tensor.shape[0]
        per_sample = adaptivity_tensor.float().reshape(batch_size, -1)
        centroid = per_sample.mean(dim=0)
        centered = per_sample - centroid
        cross_sample_std = centered.pow(2).mean().sqrt()
        adaptivity_ratio = centered.norm() / per_sample.norm().clamp_min(1e-12)
        module.log(f"{prefix}/cross_sample_std", cross_sample_std)
        module.log(f"{prefix}/adaptivity_ratio", adaptivity_ratio)
        module.log(
            f"{prefix}/centroid_cosine_mean",
            self.__mean_cosine_to_centroid(per_sample, centroid),
        )

    def __mean_cosine_to_centroid(
        self,
        per_sample: Tensor,
        centroid: Tensor,
    ) -> Tensor:
        normalized_samples = F.normalize(per_sample, dim=1)
        normalized_centroid = F.normalize(centroid, dim=0)
        return (normalized_samples @ normalized_centroid).mean()

    def __log_base_and_delta_stats(
        self,
        module: "LightningModule",
        prefix: str,
        base_tensor: Tensor,
        delta_tensor: Tensor,
    ) -> None:
        base_values = base_tensor.float()
        delta_values = delta_tensor.float()
        module.log(f"{prefix}/base_mean", base_values.mean())
        module.log(f"{prefix}/base_var", base_values.var(unbiased=False))
        module.log(f"{prefix}/delta_mean", delta_values.mean())
        module.log(f"{prefix}/delta_var", delta_values.var(unbiased=False))
        module.log(f"{prefix}/delta_l2_norm", delta_values.norm())
        module.log(
            f"{prefix}/relative_delta_norm",
            delta_values.norm() / base_values.norm().clamp_min(1e-6),
        )

    def __log_internal_stats(
        self,
        module: "LightningModule",
        prefix: str,
        slot: str,
        option: "Module",
        base_tensor: Tensor | None,
        output_tensor: Tensor,
    ) -> None:
        if slot == "weight":
            self.__log_weight_internal_stats(module, prefix, option)
        elif slot == "bias":
            self.__log_bias_internal_stats(
                module, prefix, option, base_tensor, output_tensor
            )
        elif slot == "mask" and base_tensor is not None:
            self.__log_mask_stats(module, prefix, base_tensor, output_tensor)

    def __log_weight_internal_stats(
        self,
        module: "LightningModule",
        prefix: str,
        option: "Module",
    ) -> None:
        for attr_name in ("decay_step", "warmup_step", "scale", "clamp_limit"):
            value = getattr(option, attr_name, None)
            if torch.is_tensor(value):
                module.log(f"{prefix}/{attr_name}", value.detach().float().mean())
        self.__log_bank_stats(module, prefix, option)

    def __log_bias_internal_stats(
        self,
        module: "LightningModule",
        prefix: str,
        option: "Module",
        base_tensor: Tensor | None,
        output_tensor: Tensor,
    ) -> None:
        self.__log_bank_stats(module, prefix, option)
        if base_tensor is None:
            return
        if type(option).__name__ not in {
            "AffineTransformDynamicBias",
            "MultiplicativeDynamicBias",
            "SigmoidGatedDynamicBias",
            "TanhGatedDynamicBias",
        }:
            return
        base_values = base_tensor.detach().float()
        if torch.any(base_values.abs() <= 1e-6):
            return
        effective_scale = output_tensor.detach().float() / base_values
        module.log(f"{prefix}/effective_scale_mean", effective_scale.mean())
        module.log(
            f"{prefix}/effective_scale_var", effective_scale.var(unbiased=False)
        )

    def __log_bank_stats(
        self,
        module: "LightningModule",
        prefix: str,
        option: "Module",
    ) -> None:
        weight_bank = getattr(option, "weight_bank", None)
        if weight_bank is None or not torch.is_tensor(weight_bank):
            return
        bank_values = weight_bank.detach().float()
        module.log(f"{prefix}/weight_bank_mean", bank_values.mean())
        module.log(f"{prefix}/weight_bank_var", bank_values.var(unbiased=False))
        module.log(f"{prefix}/weight_bank_l2_norm", bank_values.norm())

    def __log_mask_stats(
        self,
        module: "LightningModule",
        prefix: str,
        base_tensor: Tensor,
        output_tensor: Tensor,
    ) -> None:
        base_values = base_tensor.detach().float()
        output_values = output_tensor.detach().float()
        module.log(
            f"{prefix}/relative_output_norm",
            output_values.norm() / base_values.norm().clamp_min(1e-6),
        )
        module.log(
            f"{prefix}/attenuated_fraction",
            (output_values.abs() < base_values.abs()).float().mean(),
        )
        module.log(
            f"{prefix}/near_zero_fraction",
            (output_values.abs() <= 1e-6).float().mean(),
        )

    def __log_histograms(
        self,
        module: "LightningModule",
        prefix: str,
        output_tensor: Tensor,
        delta_tensor: Tensor | None,
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return
        step = getattr(module, "global_step", 0)
        self._emission_policy.emit_histogram(
            experiment, f"{prefix}/output", output_tensor, step
        )
        if delta_tensor is not None:
            self._emission_policy.emit_histogram(
                experiment, f"{prefix}/delta", delta_tensor, step
            )

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._adaptive_modules.clear()
        self._monitored_options.clear()
        self._emission_policy.clear()
