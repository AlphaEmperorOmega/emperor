from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor

    from emperor.sampler._sampler import SamplerModel
    from emperor.sampler._usage import SamplerUsageTrackerManager


@dataclass(frozen=True)
class _SamplerUsageMetrics:
    usage_fraction: Tensor
    mass_fraction: Tensor
    active_experts: Tensor
    entropy: Tensor
    coefficient_of_variation: Tensor


@dataclass(frozen=True)
class _SamplerTrackingContext:
    pl_module: LightningModule
    module_name: str
    batch_metrics: _SamplerUsageMetrics
    cumulative_metrics: _SamplerUsageMetrics
    retention_fraction: Tensor | None
    auxiliary_loss: Tensor
    experiment: object | None
    global_step: int


class _SamplerDiagnostics:
    @staticmethod
    def calculate_usage(
        usage_counts: Tensor,
        usage_mass: Tensor,
    ) -> _SamplerUsageMetrics:
        detached_usage_counts = usage_counts.detach()
        detached_probability_mass = usage_mass.detach()
        total_expert_usage_count = detached_usage_counts.sum().clamp_min(1.0)
        expert_usage_fraction = detached_usage_counts / total_expert_usage_count
        total_probability_mass = detached_probability_mass.sum().clamp_min(1e-6)
        expert_probability_mass_fraction = (
            detached_probability_mass / total_probability_mass
        )
        log_safe_usage_fraction = expert_usage_fraction.clamp_min(1e-6)
        usage_entropy = -(log_safe_usage_fraction.log() * expert_usage_fraction).sum()
        expert_usage_standard_deviation = detached_usage_counts.std(unbiased=False)
        stabilized_mean_expert_usage_count = detached_usage_counts.mean().clamp_min(
            1e-6
        )
        usage_coefficient_of_variation = (
            expert_usage_standard_deviation / stabilized_mean_expert_usage_count
        )
        active_expert_count = (detached_usage_counts > 0).sum().float()
        return _SamplerUsageMetrics(
            usage_fraction=expert_usage_fraction,
            mass_fraction=expert_probability_mass_fraction,
            active_experts=active_expert_count,
            entropy=usage_entropy,
            coefficient_of_variation=usage_coefficient_of_variation,
        )


class SamplerMonitorCallback(Callback):
    """Log capacity and expert-usage diagnostics for sampler modules."""

    def __init__(
        self,
        log_every_n_steps: int = 100,
        history_size: int = 128,
        log_per_expert_scalars: bool = False,
    ) -> None:
        super().__init__()
        self.__validate_positive_integer("log_every_n_steps", log_every_n_steps)
        self.__validate_positive_integer("history_size", history_size)
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self.log_per_expert_scalars = log_per_expert_scalars
        self._sampler_modules: list[tuple[str, SamplerModel]] = []
        self._usage_history: dict[str, MonitorTensorHistory] = {}
        self._mass_history: dict[str, MonitorTensorHistory] = {}
        self._tracker_manager: SamplerUsageTrackerManager | None = None
        self._emission_policy = MonitorEmissionPolicy()

    @staticmethod
    def __validate_positive_integer(option_name: str, option_value: int) -> None:
        if not isinstance(option_value, int):
            raise TypeError(
                f"{option_name} must be an integer, "
                f"received {type(option_value).__name__}."
            )
        if isinstance(option_value, bool) or option_value <= 0:
            raise ValueError(
                f"{option_name} must be a positive integer, received {option_value!r}."
            )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.sampler._sampler import SamplerModel
        from emperor.sampler._usage import SamplerUsageTrackerManager

        self.__cleanup()
        self._tracker_manager = SamplerUsageTrackerManager()
        for module_name, sampler_module in pl_module.named_modules():
            if not isinstance(sampler_module, SamplerModel):
                continue
            self._tracker_manager.attach(sampler_module)
            self._sampler_modules.append((module_name, sampler_module))
            self._usage_history[module_name] = MonitorTensorHistory(self.history_size)
            self._mass_history[module_name] = MonitorTensorHistory(self.history_size)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return
        for module_name, sampler_module in self._sampler_modules:
            usage_tracker = sampler_module.usage_tracker
            if usage_tracker is None:
                continue
            batch_usage_metrics = _SamplerDiagnostics.calculate_usage(
                usage_tracker.last_expert_usage_counts,
                usage_tracker.last_expert_usage_mass,
            )
            cumulative_usage_metrics = _SamplerDiagnostics.calculate_usage(
                usage_tracker.cumulative_expert_usage_counts,
                usage_tracker.cumulative_expert_usage_mass,
            )
            tracking_context = self.__build_tracking_context(
                pl_module,
                module_name,
                sampler_module,
                batch_usage_metrics,
                cumulative_usage_metrics,
            )
            self.__track_sampler_diagnostics(tracking_context)

    @staticmethod
    def __build_tracking_context(
        pl_module: LightningModule,
        module_name: str,
        sampler_module: SamplerModel,
        batch_metrics: _SamplerUsageMetrics,
        cumulative_metrics: _SamplerUsageMetrics,
    ) -> _SamplerTrackingContext:
        updated_skip_mask = sampler_module.get_updated_skip_mask()
        sampler_auxiliary_loss = sampler_module.get_auxiliary_loss()
        retention_fraction = (
            updated_skip_mask.detach().float().mean()
            if updated_skip_mask is not None
            else None
        )
        mean_auxiliary_loss = sampler_auxiliary_loss.detach().float().mean()
        logger = getattr(pl_module, "logger", None)
        experiment = getattr(logger, "experiment", None)
        global_step = getattr(pl_module, "global_step", 0)
        tracking_context = _SamplerTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            batch_metrics=batch_metrics,
            cumulative_metrics=cumulative_metrics,
            retention_fraction=retention_fraction,
            auxiliary_loss=mean_auxiliary_loss,
            experiment=experiment,
            global_step=global_step,
        )
        return tracking_context

    def __track_sampler_diagnostics(
        self,
        context: _SamplerTrackingContext,
    ) -> None:
        self.__track_retention_fraction(context)
        self.__track_drop_fraction(context)
        self.__track_auxiliary_loss(context)
        self.__track_active_experts(context, "batch", context.batch_metrics)
        self.__track_usage_entropy(context, "batch", context.batch_metrics)
        self.__track_usage_coefficient_of_variation(
            context,
            "batch",
            context.batch_metrics,
        )
        self.__track_maximum_usage_fraction(context, "batch", context.batch_metrics)
        self.__track_minimum_usage_fraction(context, "batch", context.batch_metrics)
        self.__track_maximum_probability_mass(context, "batch", context.batch_metrics)
        self.__track_minimum_probability_mass(context, "batch", context.batch_metrics)
        self.__track_per_expert_usage_fraction(
            context,
            "batch",
            context.batch_metrics,
        )
        self.__track_per_expert_probability_mass(
            context,
            "batch",
            context.batch_metrics,
        )
        self.__track_active_experts(context, "cumulative", context.cumulative_metrics)
        self.__track_usage_entropy(context, "cumulative", context.cumulative_metrics)
        self.__track_usage_coefficient_of_variation(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_maximum_usage_fraction(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_minimum_usage_fraction(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_maximum_probability_mass(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_minimum_probability_mass(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_per_expert_usage_fraction(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_per_expert_probability_mass(
            context,
            "cumulative",
            context.cumulative_metrics,
        )
        self.__track_usage_history(context)
        self.__track_probability_mass_history(context)
        self.__track_usage_histogram(context)
        self.__track_probability_mass_histogram(context)
        self.__track_usage_heatmap(context)
        self.__track_probability_mass_heatmap(context)

    @staticmethod
    def __track_retention_fraction(context: _SamplerTrackingContext) -> None:
        if context.retention_fraction is None:
            return
        context.pl_module.log(
            f"{context.module_name}/capacity/retention_fraction",
            context.retention_fraction,
        )

    @staticmethod
    def __track_drop_fraction(context: _SamplerTrackingContext) -> None:
        if context.retention_fraction is None:
            return
        drop_fraction = 1.0 - context.retention_fraction
        context.pl_module.log(
            f"{context.module_name}/capacity/drop_fraction",
            drop_fraction,
        )

    @staticmethod
    def __track_auxiliary_loss(context: _SamplerTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/loss/auxiliary_loss",
            context.auxiliary_loss,
        )

    @staticmethod
    def __track_active_experts(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/active_experts",
            metrics.active_experts,
        )

    @staticmethod
    def __track_usage_entropy(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/usage_entropy",
            metrics.entropy,
        )

    @staticmethod
    def __track_usage_coefficient_of_variation(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/usage_coefficient_of_variation",
            metrics.coefficient_of_variation,
        )

    @staticmethod
    def __track_maximum_usage_fraction(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        maximum_expert_usage_fraction = metrics.usage_fraction.max()
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/max_usage_fraction",
            maximum_expert_usage_fraction,
        )

    @staticmethod
    def __track_minimum_usage_fraction(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        minimum_expert_usage_fraction = metrics.usage_fraction.min()
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/min_usage_fraction",
            minimum_expert_usage_fraction,
        )

    @staticmethod
    def __track_maximum_probability_mass(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        maximum_expert_probability_mass_fraction = metrics.mass_fraction.max()
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/max_probability_mass",
            maximum_expert_probability_mass_fraction,
        )

    @staticmethod
    def __track_minimum_probability_mass(
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        minimum_expert_probability_mass_fraction = metrics.mass_fraction.min()
        context.pl_module.log(
            f"{context.module_name}/{metric_scope}/min_probability_mass",
            minimum_expert_probability_mass_fraction,
        )

    def __track_per_expert_usage_fraction(
        self,
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        if not self.log_per_expert_scalars:
            return
        for expert_index, usage_fraction in enumerate(metrics.usage_fraction):
            context.pl_module.log(
                f"{context.module_name}/{metric_scope}/"
                f"expert_{expert_index}/usage_fraction",
                usage_fraction,
            )

    def __track_per_expert_probability_mass(
        self,
        context: _SamplerTrackingContext,
        metric_scope: str,
        metrics: _SamplerUsageMetrics,
    ) -> None:
        if not self.log_per_expert_scalars:
            return
        for expert_index, probability_mass_fraction in enumerate(metrics.mass_fraction):
            context.pl_module.log(
                f"{context.module_name}/{metric_scope}/"
                f"expert_{expert_index}/probability_mass",
                probability_mass_fraction,
            )

    def __track_usage_history(self, context: _SamplerTrackingContext) -> None:
        if context.experiment is None:
            return
        usage_history = self._usage_history[context.module_name]
        batch_expert_usage_fraction = context.batch_metrics.usage_fraction
        usage_history.append(batch_expert_usage_fraction)

    def __track_probability_mass_history(
        self,
        context: _SamplerTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        probability_mass_history = self._mass_history[context.module_name]
        batch_expert_probability_mass_fraction = context.batch_metrics.mass_fraction
        probability_mass_history.append(batch_expert_probability_mass_fraction)

    def __track_usage_histogram(self, context: _SamplerTrackingContext) -> None:
        if context.experiment is None:
            return
        batch_expert_usage_fraction = context.batch_metrics.usage_fraction
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/histogram/usage_fraction",
            batch_expert_usage_fraction,
            context.global_step,
        )

    def __track_probability_mass_histogram(
        self,
        context: _SamplerTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        batch_expert_probability_mass_fraction = context.batch_metrics.mass_fraction
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/histogram/probability_mass",
            batch_expert_probability_mass_fraction,
            context.global_step,
        )

    def __track_usage_heatmap(self, context: _SamplerTrackingContext) -> None:
        if context.experiment is None:
            return
        usage_history = self._usage_history[context.module_name]
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/heatmap/usage_fraction",
            usage_history,
            context.global_step,
        )

    def __track_probability_mass_heatmap(
        self,
        context: _SamplerTrackingContext,
    ) -> None:
        if context.experiment is None:
            return
        probability_mass_history = self._mass_history[context.module_name]
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/heatmap/probability_mass",
            probability_mass_history,
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
            for _, sampler_module in self._sampler_modules:
                self._tracker_manager.detach(sampler_module)
        self._tracker_manager = None
        self._sampler_modules.clear()
        self._usage_history.clear()
        self._mass_history.clear()
        self._emission_policy.clear()
