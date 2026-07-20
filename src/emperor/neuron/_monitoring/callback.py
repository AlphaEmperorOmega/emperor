from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from lightning.pytorch.callbacks import Callback

from emperor.monitoring import (
    MonitorEmissionPolicy,
    MonitorTensorHistory,
)
from emperor.neuron._monitoring.diagnostics import (
    _EntryRoutingMetrics,
    _NeuronDiagnostics,
    _NeuronObservation,
    _RouteDiagnosticMetrics,
)

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch import Tensor

    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._trace import NeuronClusterTrace


@dataclass(frozen=True)
class _NeuronTrackingContext:
    pl_module: LightningModule
    module_name: str
    cluster: NeuronCluster
    neuron_count: float
    capacity: float
    growth_events: float
    pruning_events: float
    growth_pressure: Tensor | None
    pruning_pressure: Tensor | None
    observation: _NeuronObservation | None
    route_metrics: _RouteDiagnosticMetrics | None
    entry_routing_metrics: _EntryRoutingMetrics | None
    experiment: object | None
    global_step: int


@dataclass(frozen=True)
class _ForwardReplacement:
    cluster: NeuronCluster
    original_instance_forward: object | None
    had_instance_forward: bool

    def restore(self) -> None:
        if self.had_instance_forward:
            self.cluster.__dict__["forward"] = self.original_instance_forward
        elif "forward" in self.cluster.__dict__:
            del self.cluster.__dict__["forward"]


class NeuronClusterMonitorCallback(Callback):
    """Log structural, routing, and plasticity diagnostics for neuron clusters."""

    _OWNER_ATTRIBUTE = "_emperor_neuron_monitor_callback_owner"

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
        self._clusters: list[tuple[str, NeuronCluster]] = []
        self._forward_replacements: list[_ForwardReplacement] = []
        self._latest_observations: dict[str, _NeuronObservation] = {}
        self._latest_observation_steps: dict[str, int] = {}
        self._survival_history: dict[str, MonitorTensorHistory] = {}
        self._previous_neuron_names: dict[str, set[str]] = {}
        self._pending_growth_events: dict[str, int] = {}
        self._pending_pruning_events: dict[str, int] = {}
        self._emission_policy = MonitorEmissionPolicy()
        self._last_emitted_step = 0

    @staticmethod
    def __validate_positive(option_name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{option_name} must be a positive integer, "
                f"received {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(f"{option_name} must be greater than 0.")

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        from emperor.neuron._cluster.model import NeuronCluster

        self.__cleanup()
        self._last_emitted_step = int(getattr(pl_module, "global_step", 0))
        try:
            self._clusters.extend(
                (module_name, cluster)
                for module_name, cluster in pl_module.named_modules()
                if isinstance(cluster, NeuronCluster)
            )
            for module_name, cluster in self._clusters:
                self._survival_history[module_name] = MonitorTensorHistory(
                    self.history_size,
                    normalization="unit_interval",
                )
                self._previous_neuron_names[module_name] = set(cluster.cluster)
                self._pending_growth_events[module_name] = 0
                self._pending_pruning_events[module_name] = 0
                if cluster.beam_width == 1:
                    self.__wrap_cluster_forward(module_name, cluster, pl_module)
        except Exception:
            self.__cleanup()
            raise

    def __wrap_cluster_forward(
        self,
        module_name: str,
        cluster: NeuronCluster,
        pl_module: LightningModule,
    ) -> None:
        current_monitor_owner = cluster.__dict__.get(self._OWNER_ATTRIBUTE)
        if current_monitor_owner is not None and current_monitor_owner is not self:
            raise RuntimeError(
                "NeuronCluster is already monitored by another "
                "NeuronClusterMonitorCallback."
            )
        original_forward = cluster.forward
        had_instance_forward = "forward" in cluster.__dict__
        original_instance_forward = cluster.__dict__.get("forward")

        def monitored_forward(
            input: Tensor,
            return_trace: bool = False,
        ) -> object:
            if not self.__is_capture_step(pl_module):
                self._latest_observations.pop(module_name, None)
                self._latest_observation_steps.pop(module_name, None)
                return original_forward(input, return_trace=return_trace)
            traced_output = cast(
                "tuple[Tensor, Tensor, NeuronClusterTrace]",
                original_forward(input, return_trace=True),
            )
            output, auxiliary_loss, trace = traced_output
            self._latest_observations[module_name] = _NeuronObservation(
                trace=trace,
                auxiliary_loss=auxiliary_loss.detach(),
            )
            self._latest_observation_steps[module_name] = (
                int(getattr(pl_module, "global_step", 0)) + 1
            )
            if return_trace:
                return traced_output
            return output, auxiliary_loss

        cluster.__dict__[self._OWNER_ATTRIBUTE] = self
        cluster.forward = monitored_forward
        self._forward_replacements.append(
            _ForwardReplacement(
                cluster=cluster,
                original_instance_forward=original_instance_forward,
                had_instance_forward=had_instance_forward,
            )
        )

    def __is_capture_step(self, pl_module: LightningModule) -> bool:
        if not pl_module.training:
            return False
        global_step = getattr(pl_module, "global_step", 0)
        return (global_step + 1) % self.log_every_n_steps == 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self.__observe_topology_events()
        global_step = getattr(pl_module, "global_step", 0)
        if (
            global_step <= self._last_emitted_step
            or global_step % self.log_every_n_steps != 0
        ):
            return
        for module_name, cluster in self._clusters:
            context = self.__build_tracking_context(
                pl_module,
                module_name,
                cluster,
            )
            self.__track_neuron_cluster_diagnostics(context)
            self._pending_growth_events[module_name] = 0
            self._pending_pruning_events[module_name] = 0
        self._last_emitted_step = global_step

    def __observe_topology_events(self) -> None:
        for module_name, cluster in self._clusters:
            current_neuron_names = set(cluster.cluster)
            previous_neuron_names = self._previous_neuron_names.get(
                module_name,
                current_neuron_names,
            )
            self._pending_growth_events[module_name] = self._pending_growth_events.get(
                module_name, 0
            ) + len(current_neuron_names - previous_neuron_names)
            self._pending_pruning_events[module_name] = (
                self._pending_pruning_events.get(module_name, 0)
                + len(previous_neuron_names - current_neuron_names)
            )
            self._previous_neuron_names[module_name] = current_neuron_names

    def __build_tracking_context(
        self,
        pl_module: LightningModule,
        module_name: str,
        cluster: NeuronCluster,
    ) -> _NeuronTrackingContext:
        neuron_count = float(len(cluster.cluster))
        capacity = float(
            cluster.x_axis_total_neurons
            * cluster.y_axis_total_neurons
            * cluster.z_axis_total_neurons
        )
        observation_step = self._latest_observation_steps.get(module_name)
        observation = (
            self._latest_observations.pop(module_name, None)
            if observation_step == getattr(pl_module, "global_step", 0)
            else None
        )
        if observation is not None:
            self._latest_observation_steps.pop(module_name, None)
        route_metrics = (
            _NeuronDiagnostics.calculate_route(observation.trace)
            if observation is not None
            else None
        )
        return _NeuronTrackingContext(
            pl_module=pl_module,
            module_name=module_name,
            cluster=cluster,
            neuron_count=neuron_count,
            capacity=capacity,
            growth_events=float(self._pending_growth_events.get(module_name, 0)),
            pruning_events=float(self._pending_pruning_events.get(module_name, 0)),
            growth_pressure=self.__counter_pressure(
                cluster,
                "batch_counter",
                cluster.growth_threshold,
            ),
            pruning_pressure=self.__counter_pressure(
                cluster,
                "atrophy_counter",
                getattr(cluster, "pruning_threshold", None),
            ),
            observation=observation,
            route_metrics=route_metrics,
            entry_routing_metrics=(
                _NeuronDiagnostics.calculate_entry_routing(observation.trace)
                if observation is not None
                else None
            ),
            experiment=getattr(
                getattr(pl_module, "logger", None),
                "experiment",
                None,
            ),
            global_step=getattr(pl_module, "global_step", 0),
        )

    def __track_neuron_cluster_diagnostics(
        self,
        context: _NeuronTrackingContext,
    ) -> None:
        self.__track_neuron_count(context)
        self.__track_cluster_capacity(context)
        self.__track_cluster_fill_fraction(context)
        self.__track_growth_events(context)
        self.__track_pruning_events(context)
        self.__track_growth_pressure_mean(context)
        self.__track_growth_pressure_maximum(context)
        self.__track_total_growths(context)
        self.__track_growth_budget_remaining(context)
        self.__track_growth_cooldown_remaining(context)
        self.__track_pruning_pressure_mean(context)
        self.__track_pruning_pressure_maximum(context)
        self.__track_auxiliary_loss(context)
        self.__track_route_depth_mean(context)
        self.__track_route_depth_maximum(context)
        self.__track_recurrent_steps(context)
        self.__track_escape_fraction(context)
        self.__track_valid_fraction(context)
        self.__track_halted_fraction(context)
        self.__track_active_neuron_count(context)
        self.__track_entry_routing_entropy(context)
        self.__track_marginal_entry_routing_entropy(context)
        self.__track_routing_coefficient_of_variation(context)
        self.__track_survival_history(context)
        self.__track_route_depth_histogram(context)
        self.__track_survival_heatmap(context)
        self.__track_neuron_utilization_heatmap(context)

    @staticmethod
    def __track_neuron_count(context: _NeuronTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/cluster/neuron_count",
            context.neuron_count,
        )

    @staticmethod
    def __track_cluster_capacity(context: _NeuronTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/cluster/capacity",
            context.capacity,
        )

    @staticmethod
    def __track_cluster_fill_fraction(context: _NeuronTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/cluster/fill_fraction",
            context.neuron_count / context.capacity if context.capacity > 0 else 0.0,
        )

    @staticmethod
    def __track_growth_events(context: _NeuronTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/cluster/growth/events",
            context.growth_events,
        )

    @staticmethod
    def __track_pruning_events(context: _NeuronTrackingContext) -> None:
        context.pl_module.log(
            f"{context.module_name}/cluster/pruning/events",
            context.pruning_events,
        )

    @staticmethod
    def __track_growth_pressure_mean(context: _NeuronTrackingContext) -> None:
        if context.growth_pressure is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/growth/pressure_mean",
            context.growth_pressure.mean(),
        )

    @staticmethod
    def __track_growth_pressure_maximum(context: _NeuronTrackingContext) -> None:
        if context.growth_pressure is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/growth/pressure_max",
            context.growth_pressure.max(),
        )

    @staticmethod
    def __track_total_growths(context: _NeuronTrackingContext) -> None:
        total_growth_count = getattr(context.cluster, "total_growth_count", None)
        if total_growth_count is not None:
            context.pl_module.log(
                f"{context.module_name}/cluster/growth/total_growths",
                float(total_growth_count.item()),
            )

    @staticmethod
    def __track_growth_budget_remaining(context: _NeuronTrackingContext) -> None:
        total_growth_count = getattr(context.cluster, "total_growth_count", None)
        if total_growth_count is not None:
            context.pl_module.log(
                f"{context.module_name}/cluster/growth/budget_remaining",
                float(context.cluster.max_total_growths - int(total_growth_count)),
            )

    @staticmethod
    def __track_growth_cooldown_remaining(context: _NeuronTrackingContext) -> None:
        forwards_since_last_growth = getattr(
            context.cluster,
            "forwards_since_last_growth",
            None,
        )
        if forwards_since_last_growth is not None:
            remaining_cooldown = max(
                context.cluster.growth_cooldown_steps - int(forwards_since_last_growth),
                0,
            )
            context.pl_module.log(
                f"{context.module_name}/cluster/growth/cooldown_remaining",
                float(remaining_cooldown),
            )

    @staticmethod
    def __track_pruning_pressure_mean(context: _NeuronTrackingContext) -> None:
        if context.pruning_pressure is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/pruning/pressure_mean",
            context.pruning_pressure.mean(),
        )

    @staticmethod
    def __track_pruning_pressure_maximum(context: _NeuronTrackingContext) -> None:
        if context.pruning_pressure is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/pruning/pressure_max",
            context.pruning_pressure.max(),
        )

    @staticmethod
    def __counter_pressure(
        cluster: NeuronCluster,
        counter_name: str,
        threshold: int | None,
    ) -> Tensor | None:
        if not threshold:
            return None
        counter_values = [
            float(getattr(neuron, counter_name).item())
            for neuron in cluster.cluster.values()
            if hasattr(neuron, counter_name)
        ]
        if not counter_values:
            return None
        return torch.tensor(counter_values) / float(threshold)

    @staticmethod
    def __track_auxiliary_loss(context: _NeuronTrackingContext) -> None:
        if context.observation is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/loss/auxiliary_loss",
            context.observation.auxiliary_loss.float().mean(),
        )

    @staticmethod
    def __track_route_depth_mean(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/depth_mean",
            context.route_metrics.route_depth.mean(),
        )

    @staticmethod
    def __track_route_depth_maximum(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/depth_max",
            context.route_metrics.route_depth.max(),
        )

    @staticmethod
    def __track_recurrent_steps(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/recurrent_steps",
            context.route_metrics.recurrent_steps,
        )

    @staticmethod
    def __track_escape_fraction(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/escape_fraction",
            context.route_metrics.escape_fraction,
        )

    @staticmethod
    def __track_valid_fraction(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/valid_fraction",
            context.route_metrics.valid_fraction,
        )

    @staticmethod
    def __track_halted_fraction(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/halted_fraction",
            context.route_metrics.halted_fraction,
        )

    @staticmethod
    def __track_active_neuron_count(context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/route/active_neuron_count",
            context.route_metrics.active_neuron_count,
        )

    @staticmethod
    def __track_entry_routing_entropy(context: _NeuronTrackingContext) -> None:
        if context.entry_routing_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/entry/routing_entropy",
            context.entry_routing_metrics.mean_entropy,
        )

    @staticmethod
    def __track_marginal_entry_routing_entropy(
        context: _NeuronTrackingContext,
    ) -> None:
        if context.entry_routing_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/entry/routing_entropy_marginal",
            context.entry_routing_metrics.marginal_entropy,
        )

    @staticmethod
    def __track_routing_coefficient_of_variation(
        context: _NeuronTrackingContext,
    ) -> None:
        if context.entry_routing_metrics is None:
            return
        context.pl_module.log(
            f"{context.module_name}/cluster/entry/routing_coefficient_of_variation",
            context.entry_routing_metrics.coefficient_of_variation,
        )

    def __track_survival_history(self, context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None or context.experiment is None:
            return
        self._survival_history[context.module_name].append(
            context.route_metrics.survival
        )

    def __track_route_depth_histogram(self, context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None or context.experiment is None:
            return
        self._emission_policy.emit_histogram(
            context.experiment,
            f"{context.module_name}/cluster/histogram/route_depth",
            context.route_metrics.route_depth,
            context.global_step,
        )

    def __track_survival_heatmap(self, context: _NeuronTrackingContext) -> None:
        if context.route_metrics is None or context.experiment is None:
            return
        self._emission_policy.emit_history_heatmap(
            context.experiment,
            f"{context.module_name}/cluster/heatmap/survival",
            self._survival_history[context.module_name],
            context.global_step,
        )

    def __track_neuron_utilization_heatmap(
        self,
        context: _NeuronTrackingContext,
    ) -> None:
        if context.observation is None or context.experiment is None:
            return
        utilization_grid = torch.zeros(
            context.cluster.x_axis_total_neurons,
            context.cluster.y_axis_total_neurons,
        )
        for coordinates, valid_mask in _NeuronDiagnostics.valid_coordinates(
            context.observation.trace
        ):
            self.__accumulate_coordinate_counts(
                utilization_grid,
                coordinates,
                valid_mask,
            )
        normalized_grid = utilization_grid / utilization_grid.max().clamp_min(1e-6)
        self._emission_policy.emit_image(
            context.experiment,
            f"{context.module_name}/cluster/heatmap/neuron_utilization",
            normalized_grid.unsqueeze(0),
            context.global_step,
            dataformats="CHW",
        )

    @staticmethod
    def __accumulate_coordinate_counts(
        utilization_grid: Tensor,
        coordinates: Tensor,
        valid_mask: Tensor,
    ) -> None:
        valid_coordinates = (
            coordinates.detach()[valid_mask.detach().bool()].reshape(-1, 3).cpu()
        )
        if valid_coordinates.numel() == 0:
            return
        x_indices = valid_coordinates[:, 0].long() - 1
        y_indices = valid_coordinates[:, 1].long() - 1
        coordinates_in_range = (
            (x_indices >= 0)
            & (x_indices < utilization_grid.shape[0])
            & (y_indices >= 0)
            & (y_indices < utilization_grid.shape[1])
        )
        x_indices = x_indices[coordinates_in_range]
        y_indices = y_indices[coordinates_in_range]
        if x_indices.numel() == 0:
            return
        flat_indices = x_indices * utilization_grid.shape[1] + y_indices
        utilization_grid.view(-1).index_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices, dtype=utilization_grid.dtype),
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
        for replacement in reversed(self._forward_replacements):
            replacement.restore()
            if replacement.cluster.__dict__.get(self._OWNER_ATTRIBUTE) is self:
                del replacement.cluster.__dict__[self._OWNER_ATTRIBUTE]
        self._forward_replacements.clear()
        self._clusters.clear()
        self._latest_observations.clear()
        self._latest_observation_steps.clear()
        self._survival_history.clear()
        self._previous_neuron_names.clear()
        self._pending_growth_events.clear()
        self._pending_pruning_events.clear()
        self._emission_policy.clear()
        self._last_emitted_step = 0
