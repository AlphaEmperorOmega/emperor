from dataclasses import dataclass

import torch
from torch import Tensor

from emperor.nn import Module


@dataclass(frozen=True)
class _GrowthCounterBaseline:
    neuron_names: tuple[str, ...]
    batch_counters: Tensor
    escape_counts: Tensor | None


class _NeuronClusterPlasticityMixin:
    def _capture_growth_counter_baseline(self) -> _GrowthCounterBaseline | None:
        if self.growth_threshold is None:
            return None
        neuron_names = tuple(sorted(self.cluster.keys()))
        return _GrowthCounterBaseline(
            neuron_names=neuron_names,
            batch_counters=torch.stack(
                [self.cluster[name].batch_counter for name in neuron_names]
            ).clone(),
            escape_counts=(
                None if self.escape_counts is None else self.escape_counts.clone()
            ),
        )

    def _mark_growth_counters_global_after_load(self, _module, _incompatible_keys):
        self._growth_counters_are_global = True

    def _check_neuron_growth(
        self,
        growth_counter_baseline: _GrowthCounterBaseline | None,
    ) -> None:
        if self.growth_threshold is None:
            return

        synchronized_batch_counters = self.__synchronize_batch_counters_across_ranks(
            growth_counter_baseline
        )
        synchronized_escape_counts = self.__synchronize_escape_counts_across_ranks(
            growth_counter_baseline
        )
        if self.__is_distributed_training_initialized():
            self._growth_counters_are_global = True

        if self.__has_exhausted_growth_budget():
            return
        if self.__is_within_growth_cooldown_after_counting_forward():
            return

        for (
            saturated_neuron_name,
            saturated_neuron,
        ) in self.__saturated_neurons_by_descending_counter(
            synchronized_batch_counters
        ):
            growth_position = self.__find_closest_empty_connection(
                saturated_neuron_name,
                synchronized_escape_counts,
            )
            if growth_position is None:
                continue

            x, y, z = growth_position
            grown_neuron_name = self._neuron_name(x, y, z)
            self._add_neuron(
                self.cluster,
                grown_neuron_name,
                self.__initialize_grown_neuron_with_synchronized_rng(
                    x,
                    y,
                    z,
                    saturated_neuron,
                ),
            )
            self.__start_grown_neuron_warmup(self.cluster[grown_neuron_name])
            self.__reset_escape_count(growth_position)
            self.__record_successful_growth()
            self._neurons_called_this_forward.add(grown_neuron_name)
            saturated_neuron.batch_counter.zero_()
            return

    def __start_grown_neuron_warmup(self, neuron: Module) -> None:
        if self.growth_warmup_steps is None:
            return
        neuron.register_buffer(
            "warmup_remaining_steps",
            torch.tensor(
                self.growth_warmup_steps,
                dtype=torch.int64,
                device=neuron.batch_counter.device,
            ),
            persistent=True,
        )

    def _advance_grown_neuron_warmup(self) -> None:
        if self.growth_warmup_steps is None:
            return
        for neuron in self.cluster.values():
            warmup_remaining_steps = getattr(
                neuron,
                "warmup_remaining_steps",
                None,
            )
            if (
                warmup_remaining_steps is not None
                and int(warmup_remaining_steps.item()) > 0
            ):
                warmup_remaining_steps -= 1

    def __has_exhausted_growth_budget(self) -> bool:
        if self.total_growth_count is None:
            return False
        return int(self.total_growth_count) >= self.max_total_growths

    def __is_within_growth_cooldown_after_counting_forward(self) -> bool:
        if self.forwards_since_last_growth is None:
            return False
        self.forwards_since_last_growth += 1
        return int(self.forwards_since_last_growth) < self.growth_cooldown_steps

    def __record_successful_growth(self) -> None:
        if self.forwards_since_last_growth is not None:
            self.forwards_since_last_growth.zero_()
        if self.total_growth_count is not None:
            self.total_growth_count += 1

    def __saturated_neurons_by_descending_counter(
        self,
        synchronized_batch_counters: dict[str, int],
    ) -> list[tuple[str, Module]]:
        saturated_neurons = [
            (neuron_name, neuron)
            for neuron_name, neuron in self.cluster.items()
            if synchronized_batch_counters[neuron_name] >= self.growth_threshold
        ]
        return sorted(
            saturated_neurons,
            key=lambda named_neuron: synchronized_batch_counters[named_neuron[0]],
            reverse=True,
        )

    def __is_distributed_training_initialized(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def __synchronize_batch_counters_across_ranks(
        self,
        growth_counter_baseline: _GrowthCounterBaseline | None,
    ) -> dict[str, int]:
        sorted_neuron_names = tuple(sorted(self.cluster.keys()))
        stacked_batch_counters = torch.stack(
            [
                self.cluster[neuron_name].batch_counter
                for neuron_name in sorted_neuron_names
            ]
        )
        if self.__is_distributed_training_initialized():
            stacked_batch_counters = self.__distributed_sum_since_baseline(
                stacked_batch_counters,
                self.__batch_counter_baseline(
                    growth_counter_baseline,
                    sorted_neuron_names,
                ),
            )
            torch.distributed.all_reduce(
                stacked_batch_counters,
                op=torch.distributed.ReduceOp.SUM,
            )
            for neuron_name, counter in zip(
                sorted_neuron_names,
                stacked_batch_counters,
                strict=True,
            ):
                self.cluster[neuron_name].batch_counter.copy_(counter)
        return {
            neuron_name: int(counter.item())
            for neuron_name, counter in zip(
                sorted_neuron_names,
                stacked_batch_counters,
                strict=True,
            )
        }

    def __batch_counter_baseline(
        self,
        growth_counter_baseline: _GrowthCounterBaseline | None,
        neuron_names: tuple[str, ...],
    ) -> Tensor | None:
        if not self._growth_counters_are_global:
            return None
        if (
            growth_counter_baseline is None
            or growth_counter_baseline.neuron_names != neuron_names
        ):
            raise RuntimeError(
                "Distributed Neuron growth topology changed during a forward pass."
            )
        return growth_counter_baseline.batch_counters

    def __distributed_sum_since_baseline(
        self,
        current_counters: Tensor,
        counter_baseline: Tensor | None,
    ) -> Tensor:
        if counter_baseline is None:
            return current_counters.clone()
        rank_counter_contribution = current_counters - counter_baseline
        if torch.distributed.get_rank() == 0:
            rank_counter_contribution.add_(counter_baseline)
        return rank_counter_contribution

    def __synchronize_escape_counts_across_ranks(
        self,
        growth_counter_baseline: _GrowthCounterBaseline | None,
    ) -> Tensor | None:
        if self.escape_counts is None:
            return None
        if not self.__is_distributed_training_initialized():
            return self.escape_counts
        escape_count_baseline = None
        if self._growth_counters_are_global:
            if (
                growth_counter_baseline is None
                or growth_counter_baseline.escape_counts is None
            ):
                raise RuntimeError(
                    "Distributed Neuron escape-count state changed during a "
                    "forward pass."
                )
            escape_count_baseline = growth_counter_baseline.escape_counts
        synchronized_escape_counts = self.__distributed_sum_since_baseline(
            self.escape_counts,
            escape_count_baseline,
        )
        torch.distributed.all_reduce(
            synchronized_escape_counts,
            op=torch.distributed.ReduceOp.SUM,
        )
        self.escape_counts.copy_(synchronized_escape_counts)
        return synchronized_escape_counts

    def __find_closest_empty_connection(
        self,
        neuron_name: str,
        synchronized_escape_counts: Tensor | None,
    ) -> tuple[int, int, int] | None:
        neuron = self.cluster[neuron_name]
        connection_coordinate_rows = (
            neuron.terminal.neuron_connections.detach().cpu().tolist()
        )
        origin_x, origin_y, origin_z = self._parse_neuron_name(neuron_name)

        empty_connection_positions = []
        for connection_coordinate_row in connection_coordinate_rows:
            connection_position = self._coordinate_from_row(connection_coordinate_row)
            if not self._is_within_grid_capacity(connection_position):
                continue
            candidate_neuron_name = self._neuron_name(*connection_position)
            if candidate_neuron_name not in self.cluster:
                empty_connection_positions.append(connection_position)

        if not empty_connection_positions:
            return None

        highest_escape_position = self.__most_escaped_position(
            empty_connection_positions,
            synchronized_escape_counts,
        )
        if highest_escape_position is not None:
            return highest_escape_position

        return min(
            empty_connection_positions,
            key=lambda candidate_position: (
                abs(candidate_position[0] - origin_x)
                + abs(candidate_position[1] - origin_y)
                + abs(candidate_position[2] - origin_z)
            ),
        )

    def __most_escaped_position(
        self,
        candidate_positions: list[tuple[int, int, int]],
        synchronized_escape_counts: Tensor | None,
    ) -> tuple[int, int, int] | None:
        if synchronized_escape_counts is None:
            return None
        host_escape_counts = synchronized_escape_counts.detach().cpu()
        position_escape_counts = [
            int(host_escape_counts[x - 1, y - 1, z - 1])
            for x, y, z in candidate_positions
        ]
        maximum_escape_count = max(position_escape_counts)
        if maximum_escape_count == 0:
            return None
        return candidate_positions[position_escape_counts.index(maximum_escape_count)]

    def __reset_escape_count(self, growth_position: tuple[int, int, int]) -> None:
        if self.escape_counts is None:
            return
        x, y, z = growth_position
        self.escape_counts[x - 1, y - 1, z - 1] = 0

    def _record_escaped_missing_positions(
        self,
        escaped_positions: list[tuple[int, int, int]],
    ) -> None:
        if self.escape_counts is None or not self.training or not escaped_positions:
            return
        zero_based_coordinates = (
            torch.tensor(
                escaped_positions,
                dtype=torch.long,
                device=self.escape_counts.device,
            )
            - 1
        )
        flattened_coordinate_indices = (
            zero_based_coordinates[:, 0] * self.y_axis_total_neurons
            + zero_based_coordinates[:, 1]
        ) * self.z_axis_total_neurons + zero_based_coordinates[:, 2]
        self.escape_counts.view(-1).index_add_(
            0,
            flattened_coordinate_indices,
            torch.ones_like(flattened_coordinate_indices),
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
        rng_fork_devices = (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
        with torch.random.fork_rng(devices=rng_fork_devices):
            torch.manual_seed(shared_seed)
            return self.__initialize_grown_neuron(x, y, z, parent_neuron)

    def __broadcast_growth_seed_from_first_rank(self) -> int:
        shared_seed_container: list[int | None] = [None]
        if torch.distributed.get_rank() == 0:
            shared_seed_container[0] = int(
                torch.randint(
                    0,
                    torch.iinfo(torch.int64).max,
                    (1,),
                    dtype=torch.int64,
                ).item()
            )
        torch.distributed.broadcast_object_list(shared_seed_container, src=0)
        return shared_seed_container[0]

    def __initialize_grown_neuron(
        self,
        x: int,
        y: int,
        z: int,
        parent_neuron: Module,
    ) -> Module:
        grown_neuron = self._initialize_neuron(
            x,
            y,
            z,
            runtime_template=parent_neuron,
        )
        if not self.mitosis_initialization_flag:
            return grown_neuron
        return self.__copy_and_perturb_parent_parameters(grown_neuron, parent_neuron)

    def __copy_and_perturb_parent_parameters(
        self,
        grown_neuron: Module,
        parent_neuron: Module,
    ) -> Module:
        grown_parameters = dict(grown_neuron.named_parameters(remove_duplicate=False))
        parent_parameters = dict(parent_neuron.named_parameters(remove_duplicate=False))
        if grown_parameters.keys() != parent_parameters.keys():
            raise RuntimeError(
                "Mitosis initialization requires the grown and parent neurons "
                "to expose the same parameter roles."
            )

        copied_sources_by_grown_parameter_id: dict[int, int] = {}
        with torch.no_grad():
            for parameter_name, grown_parameter in grown_parameters.items():
                parent_parameter = parent_parameters[parameter_name]
                grown_parameter_id = id(grown_parameter)
                existing_source_parameter_id = copied_sources_by_grown_parameter_id.get(
                    grown_parameter_id
                )
                if existing_source_parameter_id is not None:
                    if existing_source_parameter_id != id(parent_parameter):
                        raise RuntimeError(
                            "Mitosis initialization cannot copy distinct parent "
                            "parameter roles into one tied grown parameter."
                        )
                    continue
                grown_parameter.copy_(parent_parameter)
                if grown_parameter.is_floating_point() or grown_parameter.is_complex():
                    statistics_source_parameter = (
                        parent_parameter
                        if parent_parameter.dtype == torch.float64
                        or parent_parameter.is_complex()
                        else parent_parameter.float()
                    )
                    parameter_standard_deviation = statistics_source_parameter.std(
                        correction=0
                    )
                    if parameter_standard_deviation > 1e-6:
                        grown_parameter.add_(
                            torch.randn_like(grown_parameter)
                            * parameter_standard_deviation.to(grown_parameter.dtype)
                            * 0.01
                        )
                copied_sources_by_grown_parameter_id[grown_parameter_id] = id(
                    parent_parameter
                )
        return grown_neuron

    def _check_neuron_atrophy(self) -> None:
        if self.pruning_threshold is None:
            return

        self.__update_atrophy_counters()
        synchronized_atrophy_counters = (
            self.__synchronize_atrophy_counters_across_ranks()
        )
        prunable_neuron_name = self.__most_atrophied_prunable_neuron(
            synchronized_atrophy_counters
        )
        if prunable_neuron_name is None:
            return
        del self.cluster[prunable_neuron_name]

    def __update_atrophy_counters(self) -> None:
        for neuron_name, neuron in self.cluster.items():
            if neuron_name in self._neurons_called_this_forward:
                neuron.atrophy_counter.zero_()
            else:
                neuron.atrophy_counter += 1

    def __synchronize_atrophy_counters_across_ranks(self) -> dict[str, int]:
        sorted_neuron_names = sorted(self.cluster.keys())
        stacked_atrophy_counters = torch.stack(
            [
                self.cluster[neuron_name].atrophy_counter
                for neuron_name in sorted_neuron_names
            ]
        )
        if self.__is_distributed_training_initialized():
            torch.distributed.all_reduce(
                stacked_atrophy_counters,
                op=torch.distributed.ReduceOp.MIN,
            )
            for neuron_name, counter in zip(
                sorted_neuron_names,
                stacked_atrophy_counters,
                strict=True,
            ):
                self.cluster[neuron_name].atrophy_counter.copy_(counter)
        return {
            neuron_name: int(counter.item())
            for neuron_name, counter in zip(
                sorted_neuron_names,
                stacked_atrophy_counters,
                strict=True,
            )
        }

    def __most_atrophied_prunable_neuron(
        self,
        synchronized_atrophy_counters: dict[str, int],
    ) -> str | None:
        entry_plane_neuron_names = self.__entry_plane_neuron_names()
        prunable_neuron_names = [
            neuron_name
            for neuron_name in self.cluster.keys()
            if neuron_name not in entry_plane_neuron_names
            and synchronized_atrophy_counters[neuron_name] >= self.pruning_threshold
        ]
        if not prunable_neuron_names:
            return None
        return max(
            sorted(prunable_neuron_names),
            key=lambda neuron_name: synchronized_atrophy_counters[neuron_name],
        )

    def __entry_plane_neuron_names(self) -> set[str]:
        return {
            self._neuron_name(int(x), int(y), int(z))
            for x, y, z in self.entry_coordinates.tolist()
        }
