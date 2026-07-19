"""Private halting monitoring callback implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

from emperor.halting._monitoring.diagnostics import (
    _HaltingDiagnosticMetrics,
    _HaltingDiagnostics,
)
from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

    from emperor.halting._base import HaltingBase
    from emperor.halting._monitoring.tracking import (
        HaltingUsageTracker,
        HaltingUsageTrackerManager,
    )


@dataclass(frozen=True)
class _HaltingTrackingContext:
    pl_module: LightningModule
    module_name: str
    metrics: _HaltingDiagnosticMetrics
    experiment: object | None
    global_step: int


class HaltingMonitorCallback(Callback):
    """Log adaptive-compute diagnostics for every halting module in a fit."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
    ) -> None:
        super().__init__()
        self.__validate_positive("log_every_n_steps", log_every_n_steps)
        self.__validate_positive("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self._halting_layers: list[tuple[str, HaltingBase]] = []
        self._survival_history: dict[str, MonitorTensorHistory] = {}
        self._tracker_manager: HaltingUsageTrackerManager | None = None
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive(option_name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{option_name} must be a positive integer, "
                f"received {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(f"{option_name} must be greater than 0.")

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        if stage == "fit":
            self.__ensure_tracking(pl_module)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.__ensure_tracking(pl_module)

    def __ensure_tracking(self, pl_module: LightningModule) -> None:
        from emperor.halting._base import HaltingBase
        from emperor.halting._monitoring.tracking import HaltingUsageTrackerManager

        if self._tracker_manager is not None:
            return
        self._tracker_manager = HaltingUsageTrackerManager()
        for module_name, halting_module in pl_module.named_modules():
            if not isinstance(halting_module, HaltingBase):
                continue
            self._tracker_manager.attach(halting_module)
            self._halting_layers.append((module_name, halting_module))
            self._survival_history[module_name] = MonitorTensorHistory(
                self.history_size,
                normalization="unit_interval",
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        for module_name, halting_module in self._halting_layers:
            tracker = getattr(halting_module, "_usage_tracker", None)
            if tracker is None:
                continue
            context = self.__build_tracking_context(pl_module, module_name, tracker)
            self.__track_halting_diagnostics(context)

    @staticmethod
    def __build_tracking_context(
        pl_module: LightningModule,
        module_name: str,
        tracker: HaltingUsageTracker,
    ) -> _HaltingTrackingContext:
        logger = pl_module.logger
        return _HaltingTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            metrics=_HaltingDiagnostics.calculate(tracker),
            experiment=getattr(logger, "experiment", None),
            global_step=int(pl_module.global_step),
        )

    def __track_halting_diagnostics(
        self,
        context: _HaltingTrackingContext,
    ) -> None:
        self.__track_ponder_cost_mean(context)
        self.__track_ponder_cost_standard_deviation(context)
        self.__track_step_count(context)
        self.__track_halted_fraction(context)
        self.__track_accumulated_halt_probability_mean(context)
        self.__track_remaining_mass_mean(context)
        self.__track_saturation_fraction(context)
        self.__track_ponder_loss(context)
        self.__track_survival_history(context)
        self.__track_survival_histogram(context)
        self.__track_ponder_cost_histogram(context)
        self.__track_survival_heatmap(context)

    @staticmethod
    def __track_ponder_cost_mean(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/depth/ponder_cost_mean",
            context.metrics.ponder_cost_mean,
        )

    @staticmethod
    def __track_ponder_cost_standard_deviation(
        context: _HaltingTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/depth/ponder_cost_std",
            context.metrics.ponder_cost_std,
        )

    @staticmethod
    def __track_step_count(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/depth/step_count",
            context.metrics.step_count,
        )

    @staticmethod
    def __track_halted_fraction(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/halt/halted_fraction",
            context.metrics.halted_fraction,
        )

    @staticmethod
    def __track_accumulated_halt_probability_mean(
        context: _HaltingTrackingContext,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/halt/accumulated_halt_prob_mean",
            context.metrics.accumulated_halt_probability_mean,
        )

    @staticmethod
    def __track_remaining_mass_mean(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/halt/remaining_mass_mean",
            context.metrics.remaining_mass_mean,
        )

    @staticmethod
    def __track_saturation_fraction(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/halt/saturation_fraction",
            context.metrics.final_survival_fraction,
        )

    @staticmethod
    def __track_ponder_loss(context: _HaltingTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/loss/ponder_loss",
            context.metrics.ponder_loss,
        )

    def __track_survival_history(self, context: _HaltingTrackingContext) -> None:
        if context.experiment is None:
            return
        self._survival_history[context.module_name].append(context.metrics.survival)

    def __track_survival_histogram(self, context: _HaltingTrackingContext) -> None:
        if context.experiment is None or context.metrics.survival.numel() == 0:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/histogram/survival",
            context.metrics.survival,
            context.global_step,
        )

    def __track_ponder_cost_histogram(
        self,
        context: _HaltingTrackingContext,
    ) -> None:
        if context.experiment is None or context.metrics.ponder_cost.numel() == 0:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/histogram/ponder_cost",
            context.metrics.ponder_cost,
            context.global_step,
        )

    def __track_survival_heatmap(self, context: _HaltingTrackingContext) -> None:
        if context.experiment is None:
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/heatmap/survival",
            self._survival_history[context.module_name],
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
        if self._tracker_manager is not None:
            for _, halting_module in self._halting_layers:
                self._tracker_manager.detach(halting_module)
        self._tracker_manager = None
        self._halting_layers.clear()
        self._survival_history.clear()
        self._emission_policy.clear()
