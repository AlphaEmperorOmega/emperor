import torch

from lightning.pytorch.callbacks import Callback
from emperor.experiments.monitor_policy import MonitorEmissionPolicy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from lightning import LightningModule, Trainer
    from emperor.sampler.model import SamplerModel


class SamplerMonitorCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_expert_scalars: bool = False,
    ):
        super().__init__()
        self.__validate_positive_integer("log_every_n_steps", log_every_n_steps)
        self.__validate_positive_integer("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_expert_scalars = log_per_expert_scalars
        self._sampler_modules = []
        self._usage_history = {}
        self._mass_history = {}
        self._tracker_manager = None
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive_integer(name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, received {type(value).__name__}."
            )
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.sampler.model import SamplerModel
        from emperor.sampler.core.tracker import SamplerUsageTrackerManager

        self.__clear_tracking_state()
        self._emission_policy.clear()
        self._tracker_manager = SamplerUsageTrackerManager()
        for name, module in pl_module.named_modules():
            if isinstance(module, SamplerModel):
                self._tracker_manager.attach(module)
                self._sampler_modules.append((name, module))
                self._usage_history[name] = []
                self._mass_history[name] = []

    def __clear_tracking_state(self) -> None:
        if self._tracker_manager is not None:
            for _, sampler in self._sampler_modules:
                self._tracker_manager.detach(sampler)
        self._tracker_manager = None
        self._sampler_modules.clear()
        self._usage_history.clear()
        self._mass_history.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return

        for name, sampler in self._sampler_modules:
            usage_tracker = sampler.usage_tracker
            if usage_tracker is None:
                continue
            self.__log_capacity_metrics(pl_module, name, sampler)
            self.__log_auxiliary_loss(pl_module, name, sampler)
            self.__log_usage_stats(
                pl_module,
                name,
                usage_tracker.last_expert_usage_counts.detach(),
                usage_tracker.last_expert_usage_mass.detach(),
                "batch",
            )
            self.__log_usage_stats(
                pl_module,
                name,
                usage_tracker.cumulative_expert_usage_counts.detach(),
                usage_tracker.cumulative_expert_usage_mass.detach(),
                "cumulative",
            )
            self.__log_visual_summaries(
                pl_module,
                name,
                usage_tracker.last_expert_usage_counts.detach(),
                usage_tracker.last_expert_usage_mass.detach(),
            )

    def __log_capacity_metrics(
        self,
        module: "LightningModule",
        name: str,
        sampler: "SamplerModel",
    ) -> None:
        skip_mask = sampler.get_updated_skip_mask()
        if skip_mask is None:
            return
        retention_fraction = skip_mask.detach().float().mean()
        module.log(f"{name}/capacity/retention_fraction", retention_fraction)
        module.log(f"{name}/capacity/drop_fraction", 1.0 - retention_fraction)

    def __log_auxiliary_loss(
        self,
        module: "LightningModule",
        name: str,
        sampler: "SamplerModel",
    ) -> None:
        auxiliary_loss = sampler.get_auxiliary_loss()
        if auxiliary_loss is None:
            return
        module.log(
            f"{name}/loss/auxiliary_loss",
            auxiliary_loss.detach().float().mean(),
        )

    def __log_usage_stats(
        self,
        module: "LightningModule",
        name: str,
        usage_counts: "Tensor",
        usage_mass: "Tensor",
        prefix: str,
    ) -> None:
        total_count = usage_counts.sum().clamp_min(1.0)
        total_mass = usage_mass.sum().clamp_min(1e-6)
        usage_fraction = usage_counts / total_count
        mass_fraction = usage_mass / total_mass
        active_experts = (usage_counts > 0).sum().float()
        entropy = -(usage_fraction.clamp_min(1e-6).log() * usage_fraction).sum()
        coefficient_of_variation = usage_counts.std() / usage_counts.mean().clamp_min(
            1e-6
        )

        module.log(f"{name}/{prefix}/active_experts", active_experts)
        module.log(f"{name}/{prefix}/usage_entropy", entropy)
        module.log(
            f"{name}/{prefix}/usage_coefficient_of_variation",
            coefficient_of_variation,
        )
        module.log(f"{name}/{prefix}/max_usage_fraction", usage_fraction.max())
        module.log(f"{name}/{prefix}/min_usage_fraction", usage_fraction.min())
        module.log(f"{name}/{prefix}/max_probability_mass", mass_fraction.max())
        module.log(f"{name}/{prefix}/min_probability_mass", mass_fraction.min())

        if self.log_per_expert_scalars:
            for expert_idx, value in enumerate(usage_fraction):
                module.log(f"{name}/{prefix}/expert_{expert_idx}/usage_fraction", value)
            for expert_idx, value in enumerate(mass_fraction):
                module.log(
                    f"{name}/{prefix}/expert_{expert_idx}/probability_mass", value
                )

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        usage_counts: "Tensor",
        usage_mass: "Tensor",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return

        step = getattr(module, "global_step", 0)
        usage_fraction = usage_counts / usage_counts.sum().clamp_min(1.0)
        mass_fraction = usage_mass / usage_mass.sum().clamp_min(1e-6)
        self.__append_history(self._usage_history[name], usage_fraction)
        self.__append_history(self._mass_history[name], mass_fraction)
        self.__log_histogram(
            experiment,
            f"{name}/histogram/usage_fraction",
            usage_fraction,
            step,
        )
        self.__log_histogram(
            experiment,
            f"{name}/histogram/probability_mass",
            mass_fraction,
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/heatmap/usage_fraction",
            self._usage_history[name],
            step,
        )
        self.__log_heatmap(
            experiment,
            f"{name}/heatmap/probability_mass",
            self._mass_history[name],
            step,
        )

    def __append_history(self, history: list["Tensor"], values: "Tensor") -> None:
        history.append(values.detach().float().cpu())
        del history[: -self.history_size]

    def __log_histogram(
        self, experiment, tag: str, values: "Tensor", step: int
    ) -> None:
        self._emission_policy.emit_histogram(experiment, tag, values, step)

    def __log_heatmap(
        self,
        experiment,
        tag: str,
        history: list["Tensor"],
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
        self.__clear_tracking_state()
        self._emission_policy.clear()
