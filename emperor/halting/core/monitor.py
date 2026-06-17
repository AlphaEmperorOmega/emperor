import torch

from lightning.pytorch.callbacks import Callback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from lightning import LightningModule, Trainer
    from emperor.halting.core.tracker import HaltingUsageTracker


class HaltingMonitorCallback(Callback):
    """Logs adaptive-compute dynamics for every halting module.

    Capture is installed at runtime (the tracker manager wraps the halting
    methods) and removed on fit end, so no traced code is modified. Works for any
    halting owner (recurrent layer or neuron cluster). Emits depth / halting /
    ponder-loss scalars plus a survival heatmap (recurrence step x training time)
    per halting module.
    """

    def __init__(self, log_every_n_steps: int = 100, history_size: int = 128):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self._halting_layers = []
        self._survival_history = {}
        self._tracker_manager = None

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.halting.core.base import HaltingBase
        from emperor.halting.core.tracker import HaltingUsageTrackerManager

        self._tracker_manager = HaltingUsageTrackerManager()
        halting_modules = [
            (name, module)
            for name, module in pl_module.named_modules()
            if isinstance(module, HaltingBase)
        ]
        for name, module in halting_modules:
            self._tracker_manager.attach(module)
            self._halting_layers.append((name, module))
            self._survival_history[name] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        for name, halting_model in self._halting_layers:
            tracker = getattr(halting_model, "_usage_tracker", None)
            if tracker is None:
                continue
            self.__log_scalar_stats(pl_module, name, tracker)
            self.__log_visual_summaries(pl_module, name, tracker)

    def __log_scalar_stats(
        self,
        module: "LightningModule",
        name: str,
        tracker: "HaltingUsageTracker",
    ) -> None:
        module.log(
            f"{name}/depth/ponder_cost_mean",
            tracker.last_ponder_cost_mean.detach().float(),
        )
        module.log(
            f"{name}/depth/ponder_cost_std",
            tracker.last_ponder_cost_std.detach().float(),
        )
        module.log(
            f"{name}/depth/step_count",
            tracker.last_step_count.detach().float(),
        )
        module.log(
            f"{name}/halt/halted_fraction",
            tracker.last_halted_fraction.detach().float(),
        )
        module.log(
            f"{name}/halt/accumulated_halt_prob_mean",
            tracker.last_accumulated_halt_prob_mean.detach().float(),
        )
        module.log(
            f"{name}/halt/remaining_mass_mean",
            tracker.last_remaining_mass_mean.detach().float(),
        )
        module.log(
            f"{name}/loss/ponder_loss",
            tracker.last_ponder_loss.detach().float(),
        )

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        tracker: "HaltingUsageTracker",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return

        step = getattr(module, "global_step", 0)
        survival = tracker.last_survival.detach().float().cpu()
        self.__append_history(self._survival_history[name], survival)
        self.__log_histogram(experiment, f"{name}/histogram/survival", survival, step)
        self.__log_heatmap(
            experiment,
            f"{name}/heatmap/survival",
            self._survival_history[name],
            step,
        )

    def __append_history(self, history: list["Tensor"], values: "Tensor") -> None:
        history.append(values.detach().float().cpu())
        del history[: -self.history_size]

    def __log_histogram(
        self,
        experiment,
        tag: str,
        values: "Tensor",
        step: int,
    ) -> None:
        if hasattr(experiment, "add_histogram"):
            experiment.add_histogram(tag, values.detach().float().cpu(), step)

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
            torch.nn.functional.pad(vector, (0, max_steps - vector.numel()))
            for vector in history
        ]
        heatmap = torch.stack(padded, dim=0).T.clamp(0.0, 1.0)
        image = heatmap.unsqueeze(0)
        experiment.add_image(tag, image, step, dataformats="CHW")

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if self._tracker_manager is not None:
            for _, halting_model in self._halting_layers:
                self._tracker_manager.detach(halting_model)
        self._tracker_manager = None
        self._halting_layers.clear()
        self._survival_history.clear()
