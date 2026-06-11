import torch

from lightning.pytorch.callbacks import Callback

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor
    from lightning import LightningModule, Trainer
    from emperor.neuron.core.model import NeuronCluster
    from emperor.neuron.core.state import NeuronClusterTrace


class NeuronClusterMonitorCallback(Callback):
    """Logs how a neuron cluster grows and routes signals over training.

    Emits scalars under each cluster module's path so the viewer surfaces them
    in the cluster node's monitor charts. Beyond capacity (``neuron_count`` vs
    ``capacity``), it captures the cluster's routing dynamics by wrapping
    ``forward`` to request its ``NeuronClusterTrace`` (entry + per recurrent
    step), entirely non-invasively: the wrapper is installed in ``on_fit_start``
    and removed in ``on_fit_end``, and the traced ``forward`` is never edited.

    From the trace it derives route depth, escape / halt fractions, entry-routing
    entropy, per-neuron spatial utilization, and a recurrence survival curve;
    growth pressure is read directly from each neuron's ``batch_counter``.
    """

    def __init__(self, log_every_n_steps: int = 100, history_size: int = 128):
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self.log_every_n_steps = log_every_n_steps
        self.history_size = history_size
        self._clusters: list[tuple[str, "NeuronCluster"]] = []
        self._latest_trace: dict[str, "NeuronClusterTrace | None"] = {}
        self._latest_loss: dict[str, "Tensor | None"] = {}
        self._survival_history: dict[str, list["Tensor"]] = {}
        self._previous_neuron_count: dict[str, int] = {}

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        from emperor.neuron.core.model import NeuronCluster

        self._clusters = [
            (name, module)
            for name, module in pl_module.named_modules()
            if isinstance(module, NeuronCluster)
        ]
        for name, cluster in self._clusters:
            self._latest_trace[name] = None
            self._latest_loss[name] = None
            self._survival_history[name] = []
            self._previous_neuron_count[name] = len(cluster.cluster)
            self.__wrap_cluster_forward(name, cluster, pl_module)

    def __wrap_cluster_forward(
        self,
        name: str,
        cluster: "NeuronCluster",
        pl_module: "LightningModule",
    ) -> None:
        original_forward = cluster.forward

        def forward(input, return_trace=False):
            if not self.__is_capture_step(pl_module):
                return original_forward(input, return_trace=return_trace)
            output, auxiliary_loss, trace = cast(
                "tuple[Tensor, Tensor, NeuronClusterTrace]",
                original_forward(input, return_trace=True),
            )
            self._latest_trace[name] = trace
            self._latest_loss[name] = auxiliary_loss.detach()
            if return_trace:
                return output, auxiliary_loss, trace
            return output, auxiliary_loss

        cluster.forward = forward

    def __is_capture_step(self, pl_module: "LightningModule") -> bool:
        if not pl_module.training:
            return False
        step = getattr(pl_module, "global_step", 0)
        return step % self.log_every_n_steps == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if getattr(pl_module, "global_step", 0) % self.log_every_n_steps != 0:
            return
        for name, cluster in self._clusters:
            self.__log_cluster_size(pl_module, name, cluster)
            self.__log_growth_pressure(pl_module, name, cluster)
            trace = self._latest_trace.get(name)
            if trace is None:
                continue
            self.__log_auxiliary_loss(pl_module, name)
            self.__log_route_dynamics(pl_module, name, trace)
            self.__log_entry_routing(pl_module, name, trace)
            self.__log_visual_summaries(pl_module, name, cluster, trace)

    def __log_cluster_size(
        self,
        module: "LightningModule",
        name: str,
        cluster: "NeuronCluster",
    ) -> None:
        neuron_count = float(len(cluster.cluster))
        capacity = float(
            cluster.x_axis_total_neurons
            * cluster.y_axis_total_neurons
            * cluster.z_axis_total_neurons
        )
        module.log(f"{name}/cluster/neuron_count", neuron_count)
        module.log(f"{name}/cluster/capacity", capacity)
        module.log(
            f"{name}/cluster/fill_fraction",
            neuron_count / capacity if capacity > 0 else 0.0,
        )

    def __log_growth_pressure(
        self,
        module: "LightningModule",
        name: str,
        cluster: "NeuronCluster",
    ) -> None:
        current_count = len(cluster.cluster)
        previous_count = self._previous_neuron_count.get(name, current_count)
        growth_events = max(0, current_count - previous_count)
        self._previous_neuron_count[name] = current_count
        module.log(f"{name}/cluster/growth/events", float(growth_events))

        growth_threshold = cluster.growth_threshold
        if not growth_threshold:
            return
        counters = [
            float(getattr(neuron, "batch_counter").item())
            for neuron in cluster.cluster.values()
            if hasattr(neuron, "batch_counter")
        ]
        if not counters:
            return
        pressure = torch.tensor(counters) / float(growth_threshold)
        module.log(f"{name}/cluster/growth/pressure_mean", pressure.mean())
        module.log(f"{name}/cluster/growth/pressure_max", pressure.max())

    def __log_auxiliary_loss(self, module: "LightningModule", name: str) -> None:
        loss = self._latest_loss.get(name)
        if loss is not None:
            module.log(f"{name}/cluster/loss/auxiliary_loss", loss.float().mean())

    def __log_route_dynamics(
        self,
        module: "LightningModule",
        name: str,
        trace: "NeuronClusterTrace",
    ) -> None:
        depth = self.__compute_route_depth(trace)
        module.log(f"{name}/cluster/route/depth_mean", depth.mean())
        module.log(f"{name}/cluster/route/depth_max", depth.max())
        module.log(
            f"{name}/cluster/route/recurrent_steps", float(len(trace.steps))
        )
        module.log(
            f"{name}/cluster/route/escape_fraction",
            self.__average_mask_fraction(
                [trace.entry_escape_mask] + [step.escape_mask for step in trace.steps]
            ),
        )
        module.log(
            f"{name}/cluster/route/valid_fraction",
            self.__average_mask_fraction(
                [trace.entry_valid_mask] + [step.valid_mask for step in trace.steps]
            ),
        )
        module.log(
            f"{name}/cluster/route/halted_fraction",
            self.__final_halt_fraction(trace),
        )
        module.log(
            f"{name}/cluster/route/active_neuron_count",
            self.__count_active_neurons(trace),
        )

    def __compute_route_depth(self, trace: "NeuronClusterTrace") -> "Tensor":
        depth = trace.entry_active_mask.detach().float()
        for step in trace.steps:
            depth = depth + step.active_mask.detach().float()
        return depth

    def __average_mask_fraction(self, masks: list["Tensor"]) -> "Tensor":
        fractions = [mask.detach().float().mean() for mask in masks if mask.numel() > 0]
        if not fractions:
            return torch.zeros(())
        return torch.stack(fractions).mean()

    def __final_halt_fraction(self, trace: "NeuronClusterTrace") -> "Tensor":
        if trace.steps:
            return trace.steps[-1].halt_mask.detach().float().mean()
        return trace.entry_halt_mask.detach().float().mean()

    def __count_active_neurons(self, trace: "NeuronClusterTrace") -> "Tensor":
        coordinate_rows = []
        for coords, valid in self.__iter_valid_coordinates(trace):
            valid_coords = coords[valid.bool()]
            if valid_coords.numel() > 0:
                coordinate_rows.append(valid_coords.reshape(-1, 3))
        if not coordinate_rows:
            return torch.zeros(())
        all_coordinates = torch.cat(coordinate_rows, dim=0)
        unique_count = torch.unique(all_coordinates, dim=0).shape[0]
        return torch.tensor(float(unique_count))

    def __iter_valid_coordinates(self, trace: "NeuronClusterTrace"):
        yield trace.entry_selected_coordinates, trace.entry_valid_mask
        for step in trace.steps:
            yield step.selected_coordinates, step.valid_mask

    def __log_entry_routing(
        self,
        module: "LightningModule",
        name: str,
        trace: "NeuronClusterTrace",
    ) -> None:
        probabilities = trace.entry_probabilities.detach().float()
        if probabilities.numel() == 0:
            return
        row_sums = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        normalized = probabilities / row_sums
        per_sample_entropy = self.__compute_distribution_entropy(normalized, dim=-1)
        marginal = normalized.mean(dim=0)
        marginal = marginal / marginal.sum().clamp_min(1e-9)
        marginal_entropy = self.__compute_distribution_entropy(marginal, dim=-1)
        coefficient_of_variation = marginal.std() / marginal.mean().clamp_min(1e-6)

        module.log(f"{name}/cluster/entry/routing_entropy", per_sample_entropy.mean())
        module.log(
            f"{name}/cluster/entry/routing_entropy_marginal", marginal_entropy
        )
        module.log(
            f"{name}/cluster/entry/routing_coefficient_of_variation",
            coefficient_of_variation,
        )

    def __compute_distribution_entropy(
        self,
        distribution: "Tensor",
        dim: int,
    ) -> "Tensor":
        safe_distribution = distribution.clamp_min(1e-9)
        return -(safe_distribution.log() * distribution).sum(dim=dim)

    def __log_visual_summaries(
        self,
        module: "LightningModule",
        name: str,
        cluster: "NeuronCluster",
        trace: "NeuronClusterTrace",
    ) -> None:
        experiment = getattr(getattr(module, "logger", None), "experiment", None)
        if experiment is None:
            return

        step = getattr(module, "global_step", 0)
        survival = self.__compute_survival_curve(trace)
        self.__append_history(self._survival_history[name], survival)
        self.__log_histogram(
            experiment,
            f"{name}/cluster/histogram/route_depth",
            self.__compute_route_depth(trace),
            step,
        )
        self.__log_padded_heatmap(
            experiment,
            f"{name}/cluster/heatmap/survival",
            self._survival_history[name],
            step,
        )
        self.__log_utilization_heatmap(experiment, name, cluster, trace, step)

    def __compute_survival_curve(self, trace: "NeuronClusterTrace") -> "Tensor":
        stages = [trace.entry_active_mask.detach().float().mean()]
        stages.extend(step.active_mask.detach().float().mean() for step in trace.steps)
        return torch.stack(stages)

    def __log_utilization_heatmap(
        self,
        experiment,
        name: str,
        cluster: "NeuronCluster",
        trace: "NeuronClusterTrace",
        step: int,
    ) -> None:
        if not hasattr(experiment, "add_image"):
            return
        grid = torch.zeros(
            cluster.x_axis_total_neurons, cluster.y_axis_total_neurons
        )
        for coords, valid in self.__iter_valid_coordinates(trace):
            self.__accumulate_coordinate_counts(grid, coords, valid)
        grid = grid / grid.max().clamp_min(1e-6)
        experiment.add_image(
            f"{name}/cluster/heatmap/neuron_utilization",
            grid.unsqueeze(0),
            step,
            dataformats="CHW",
        )

    def __accumulate_coordinate_counts(
        self,
        grid: "Tensor",
        coords: "Tensor",
        valid: "Tensor",
    ) -> None:
        valid_coords = coords.detach()[valid.detach().bool()].reshape(-1, 3)
        if valid_coords.numel() == 0:
            return
        x_indices = valid_coords[:, 0].long() - 1
        y_indices = valid_coords[:, 1].long() - 1
        in_range = (
            (x_indices >= 0)
            & (x_indices < grid.shape[0])
            & (y_indices >= 0)
            & (y_indices < grid.shape[1])
        )
        x_indices = x_indices[in_range]
        y_indices = y_indices[in_range]
        if x_indices.numel() == 0:
            return
        flat_indices = x_indices * grid.shape[1] + y_indices
        grid.view(-1).index_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=grid.dtype)
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

    def __log_padded_heatmap(
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
        experiment.add_image(tag, heatmap.unsqueeze(0), step, dataformats="CHW")

    def on_fit_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        for _, cluster in self._clusters:
            if "forward" in cluster.__dict__:
                del cluster.__dict__["forward"]
        self._clusters.clear()
        self._latest_trace.clear()
        self._latest_loss.clear()
        self._survival_history.clear()
        self._previous_neuron_count.clear()
