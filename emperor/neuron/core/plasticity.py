import torch

from torch import Tensor

from emperor.base.utils import Module


class NeuronClusterPlasticityMixin:
    """Structural plasticity for NeuronCluster: growth machinery and the
    state-dict reconciliation that keeps checkpoints loadable across
    cluster-shape changes."""

    def _check_neuron_growth(self) -> None:
        """Grows at most one neuron per training forward.

        Under an initialized process group every rank participates in the
        collectives below on every training forward, so ranks must call
        forward in lockstep (standard DDP semantics) or growth deadlocks.
        """
        if self.growth_threshold is None:
            return

        synchronized_batch_counters = (
            self.__synchronize_batch_counters_across_ranks()
        )
        synchronized_escape_counts = (
            self.__synchronize_escape_counts_across_ranks()
        )

        for name, neuron in self.__saturated_neurons_by_descending_counter(
            synchronized_batch_counters
        ):
            position = self.__find_closest_empty_connection(
                name,
                synchronized_escape_counts,
            )
            if position is None:
                continue

            neuron.batch_counter.zero_()
            x, y, z = position
            new_name = self._neuron_name(x, y, z)
            self._add_neuron(
                self.cluster,
                new_name,
                self.__initialize_grown_neuron_with_synchronized_rng(
                    x,
                    y,
                    z,
                    neuron,
                ),
            )
            self.__reset_escape_count(position)
            return

    def __saturated_neurons_by_descending_counter(
        self,
        synchronized_batch_counters: dict[str, int],
    ) -> list[tuple[str, Module]]:
        saturated_neurons = [
            (name, neuron)
            for name, neuron in self.cluster.items()
            if synchronized_batch_counters[name] >= self.growth_threshold
        ]
        return sorted(
            saturated_neurons,
            key=lambda entry: synchronized_batch_counters[entry[0]],
            reverse=True,
        )

    def __is_distributed_training_initialized(self) -> bool:
        return (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

    def __synchronize_batch_counters_across_ranks(self) -> dict[str, int]:
        sorted_neuron_names = sorted(self.cluster.keys())
        stacked_counters = torch.stack(
            [self.cluster[name].batch_counter for name in sorted_neuron_names]
        )
        if self.__is_distributed_training_initialized():
            torch.distributed.all_reduce(
                stacked_counters,
                op=torch.distributed.ReduceOp.SUM,
            )
        return {
            name: int(counter)
            for name, counter in zip(
                sorted_neuron_names,
                stacked_counters.tolist(),
            )
        }

    def __synchronize_escape_counts_across_ranks(self) -> Tensor | None:
        if self.escape_counts is None:
            return None
        if not self.__is_distributed_training_initialized():
            return self.escape_counts
        synchronized_escape_counts = self.escape_counts.clone()
        torch.distributed.all_reduce(
            synchronized_escape_counts,
            op=torch.distributed.ReduceOp.SUM,
        )
        return synchronized_escape_counts

    def __find_closest_empty_connection(
        self,
        neuron_name: str,
        synchronized_escape_counts: Tensor | None,
    ) -> tuple[int, int, int] | None:
        neuron = self.cluster[neuron_name]
        connection_rows = neuron.terminal.neuron_connections.detach().cpu().tolist()
        origin_x, origin_y, origin_z = self._parse_neuron_name(neuron_name)

        empty_positions = []
        for connection_row in connection_rows:
            position = self._coordinate_from_row(connection_row)
            if not self._is_within_grid_capacity(position):
                continue
            candidate_name = self._neuron_name(*position)
            if candidate_name not in self.cluster:
                empty_positions.append(position)

        if not empty_positions:
            return None

        escape_selected_position = self.__most_escaped_position(
            empty_positions,
            synchronized_escape_counts,
        )
        if escape_selected_position is not None:
            return escape_selected_position

        return min(
            empty_positions,
            key=lambda pos: abs(pos[0] - origin_x)
            + abs(pos[1] - origin_y)
            + abs(pos[2] - origin_z),
        )

    def __most_escaped_position(
        self,
        positions: list[tuple[int, int, int]],
        synchronized_escape_counts: Tensor | None,
    ) -> tuple[int, int, int] | None:
        if synchronized_escape_counts is None:
            return None
        counts_cpu = synchronized_escape_counts.detach().cpu()
        counts = [int(counts_cpu[x - 1, y - 1, z - 1]) for x, y, z in positions]
        max_count = max(counts)
        if max_count == 0:
            return None
        return positions[counts.index(max_count)]

    def __reset_escape_count(self, position: tuple[int, int, int]) -> None:
        if self.escape_counts is None:
            return
        x, y, z = position
        self.escape_counts[x - 1, y - 1, z - 1] = 0

    def _record_escaped_missing_positions(
        self,
        positions: list[tuple[int, int, int]],
    ) -> None:
        if self.escape_counts is None or not self.training or not positions:
            return
        coordinate_tensor = (
            torch.tensor(
                positions,
                dtype=torch.long,
                device=self.escape_counts.device,
            )
            - 1
        )
        flat_indices = (
            coordinate_tensor[:, 0] * self.y_axis_total_neurons
            + coordinate_tensor[:, 1]
        ) * self.z_axis_total_neurons + coordinate_tensor[:, 2]
        self.escape_counts.view(-1).index_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices),
        )

    def __initialize_grown_neuron_with_synchronized_rng(
        self,
        x: int,
        y: int,
        z: int,
        parent_neuron: Module,
    ) -> Module:
        if not self.__is_distributed_training_initialized():
            return self.__initialize_grown_neuron(x, y, z, parent_neuron)

        shared_seed = self.__broadcast_growth_seed_from_first_rank()
        fork_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(shared_seed)
            return self.__initialize_grown_neuron(x, y, z, parent_neuron)

    def __broadcast_growth_seed_from_first_rank(self) -> int:
        seed_container: list[int | None] = [None]
        if torch.distributed.get_rank() == 0:
            seed_container[0] = int(
                torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (1,),
                    dtype=torch.int64,
                ).item()
            )
        torch.distributed.broadcast_object_list(seed_container, src=0)
        return seed_container[0]

    def __initialize_grown_neuron(
        self,
        x: int,
        y: int,
        z: int,
        parent_neuron: Module,
    ) -> Module:
        grown_neuron = self._initialize_neuron(x, y, z)
        if not self.mitosis_initialization_flag:
            return grown_neuron
        return self.__copy_and_perturb_parent_parameters(grown_neuron, parent_neuron)

    def __copy_and_perturb_parent_parameters(
        self,
        grown_neuron: Module,
        parent_neuron: Module,
    ) -> Module:
        with torch.no_grad():
            for grown_parameter, parent_parameter in zip(
                grown_neuron.parameters(),
                parent_neuron.parameters(),
                strict=True,
            ):
                grown_parameter.copy_(parent_parameter)
                perturbation_std = parent_parameter.float().std(correction=0)
                if perturbation_std > 1e-6:
                    grown_parameter.add_(
                        torch.randn_like(grown_parameter)
                        * perturbation_std.to(grown_parameter.dtype)
                        * 0.01
                    )
        return grown_neuron

    def _reconcile_cluster_with_state_dict(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        cluster_prefix = f"{prefix}cluster."
        for key in state_dict:
            if not key.startswith(cluster_prefix):
                continue
            neuron_name = key[len(cluster_prefix):].split(".", 1)[0]
            if neuron_name in self.cluster:
                continue
            if not self._is_neuron_name(neuron_name):
                continue
            self._add_neuron(
                self.cluster,
                neuron_name,
                self._initialize_neuron(*self._parse_neuron_name(neuron_name)),
            )
