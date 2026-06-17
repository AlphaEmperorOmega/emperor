import torch

from lightning.pytorch.callbacks import Callback
from emperor.experiments.monitor_policy import MonitorEmissionPolicy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from lightning import LightningModule, Trainer


class WeightBankUtilizationMonitorCallback(Callback):
    DEAD_SLOT_UTILIZATION_FLOOR = 1e-4

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
        self._hooks = []
        self._bank_modules = []
        self._utilization_history = {}
        self._last_bank_logits = {}
        self._emission_policy = MonitorEmissionPolicy()

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._emission_policy.clear()
        for name, module in pl_module.named_modules():
            if not self.__is_weighted_bank_module(module):
                continue
            self._bank_modules.append((name, module))
            self._utilization_history[name] = []
            generator_model = getattr(module, "model")
            hook = generator_model.register_forward_hook(
                self.__make_bank_logits_capture_hook(name)
            )
            self._hooks.append(hook)

    def __is_weighted_bank_module(self, module: "Module") -> bool:
        from emperor.augmentations.adaptive_parameters.core.weight.variants import (
            LayeredWeightedBankDynamicWeight,
            SoftWeightedBankDynamicWeight,
        )
        from emperor.augmentations.adaptive_parameters.core.bias.variants import (
            WeightedBankDynamicBias,
        )

        weighted_bank_types = (
            LayeredWeightedBankDynamicWeight,
            SoftWeightedBankDynamicWeight,
            WeightedBankDynamicBias,
        )
        return isinstance(module, weighted_bank_types)

    def __make_bank_logits_capture_hook(self, name: str):
        def hook(generator: "Module", inputs: tuple, output: object) -> None:
            bank_logits = self.__extract_bank_logits(output)
            if bank_logits is not None:
                self._last_bank_logits[name] = bank_logits.detach()

        return hook

    def __extract_bank_logits(self, output: object) -> "Tensor | None":
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        if torch.is_tensor(hidden):
            return hidden
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return
        for name, module in self._bank_modules:
            bank_logits = self._last_bank_logits.get(name)
            if bank_logits is None:
                continue
            summary = self.__compute_bank_distribution_summary(module, bank_logits)
            if summary is None:
                continue
            per_slot_utilization, mean_per_sample_entropy = summary
            self.__log_bank_utilization_scalars(
                pl_module, name, per_slot_utilization, mean_per_sample_entropy
            )
            self.__log_visual_summaries(pl_module, name, per_slot_utilization)

    def __compute_bank_distribution_summary(
        self,
        module: "Module",
        bank_logits: "Tensor",
    ) -> "tuple[Tensor, Tensor] | None":
        module_type_name = type(module).__name__
        if module_type_name == "SoftWeightedBankDynamicWeight":
            return self.__summarize_soft_weighted_bank(module, bank_logits)
        if module_type_name == "LayeredWeightedBankDynamicWeight":
            return self.__summarize_layered_weighted_bank(module, bank_logits)
        if module_type_name == "WeightedBankDynamicBias":
            return self.__summarize_weighted_bank_bias(module, bank_logits)
        return None

    def __summarize_soft_weighted_bank(
        self,
        module: "Module",
        bank_logits: "Tensor",
    ) -> "tuple[Tensor, Tensor]":
        depth_value = getattr(module, "depth_value")
        input_dim = getattr(module, "input_dim")
        bank_expansion_factor = getattr(module, "bank_expansion_factor")
        reshaped_logits = bank_logits.view(
            -1, depth_value, input_dim, bank_expansion_factor
        )
        bank_distribution = torch.softmax(reshaped_logits, dim=-1)
        per_slot_utilization = bank_distribution.mean(dim=(0, 1, 2))
        per_sample_entropy = self.__compute_distribution_entropy(
            bank_distribution, dim=-1
        )
        return per_slot_utilization, per_sample_entropy.mean()

    def __summarize_layered_weighted_bank(
        self,
        module: "Module",
        bank_logits: "Tensor",
    ) -> "tuple[Tensor, Tensor]":
        depth_value = getattr(module, "depth_value")
        input_dim = getattr(module, "input_dim")
        bank_expansion_factor = getattr(module, "bank_expansion_factor")
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        reshaped_distribution = bank_distribution.view(
            -1, depth_value, input_dim, bank_expansion_factor
        )
        per_slot_utilization = reshaped_distribution.sum(dim=2).mean(dim=(0, 1))
        per_sample_entropy = self.__compute_distribution_entropy(
            bank_distribution, dim=-1
        )
        return per_slot_utilization, per_sample_entropy.mean()

    def __summarize_weighted_bank_bias(
        self,
        module: "Module",
        bank_logits: "Tensor",
    ) -> "tuple[Tensor, Tensor]":
        bank_expansion_factor = getattr(module, "bank_expansion_factor")
        bank_distribution = torch.softmax(bank_logits, dim=-1)
        flat_distribution = bank_distribution.reshape(-1, bank_expansion_factor)
        per_slot_utilization = flat_distribution.mean(dim=0)
        per_sample_entropy = self.__compute_distribution_entropy(
            flat_distribution, dim=-1
        )
        return per_slot_utilization, per_sample_entropy.mean()

    def __compute_distribution_entropy(
        self,
        distribution: "Tensor",
        dim: int,
    ) -> "Tensor":
        safe_distribution = distribution.clamp_min(1e-9)
        return -(safe_distribution.log() * distribution).sum(dim=dim)

    def __log_bank_utilization_scalars(
        self,
        module: "LightningModule",
        name: str,
        per_slot_utilization: "Tensor",
        mean_per_sample_entropy: "Tensor",
    ) -> None:
        utilization = per_slot_utilization.float()
        marginal_entropy = self.__compute_distribution_entropy(utilization, dim=-1)
        coefficient_of_variation = utilization.std() / utilization.mean().clamp_min(
            1e-6
        )
        active_slots = (utilization > self.DEAD_SLOT_UTILIZATION_FLOOR).sum().float()
        dead_slot_fraction = (
            (utilization <= self.DEAD_SLOT_UTILIZATION_FLOOR).float().mean()
        )

        module.log(f"{name}/bank/selection_entropy_marginal", marginal_entropy)
        module.log(f"{name}/bank/selection_entropy_mean", mean_per_sample_entropy)
        module.log(
            f"{name}/bank/utilization_coefficient_of_variation",
            coefficient_of_variation,
        )
        module.log(f"{name}/bank/active_slots", active_slots)
        module.log(f"{name}/bank/dead_slot_fraction", dead_slot_fraction)
        module.log(f"{name}/bank/max_utilization", utilization.max())
        module.log(f"{name}/bank/min_utilization", utilization.min())

        if self.log_per_slot_scalars:
            for slot_index, value in enumerate(utilization):
                module.log(f"{name}/bank/slot_{slot_index}/utilization", value)

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        per_slot_utilization: "Tensor",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return

        step = getattr(module, "global_step", 0)
        utilization = per_slot_utilization.detach().float().cpu()
        self.__append_history(self._utilization_history[name], utilization)
        self.__log_histogram(
            experiment, f"{name}/bank/histogram/utilization", utilization, step
        )
        self.__log_heatmap(
            experiment,
            f"{name}/bank/heatmap/utilization",
            self._utilization_history[name],
            step,
        )

    def __append_history(self, history: list, values: "Tensor") -> None:
        history.append(values.detach().float().cpu())
        del history[: -self.history_size]

    def __log_histogram(
        self,
        experiment,
        tag: str,
        values: "Tensor",
        step: int,
    ) -> None:
        self._emission_policy.emit_histogram(experiment, tag, values, step)

    def __log_heatmap(
        self,
        experiment,
        tag: str,
        history: list,
        step: int,
    ) -> None:
        if not hasattr(experiment, "add_image") or not history:
            return
        heatmap = torch.stack(history, dim=0).T
        heatmap = heatmap / heatmap.max().clamp_min(1e-6)
        image = heatmap.unsqueeze(0)
        self._emission_policy.emit_image(
            experiment, tag, image, step, dataformats="CHW"
        )

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._bank_modules.clear()
        self._utilization_history.clear()
        self._last_bank_logits.clear()
        self._emission_policy.clear()
