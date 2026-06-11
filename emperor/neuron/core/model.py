import copy

from dataclasses import fields
import torch

from torch import Tensor
from torch.nn import ModuleDict

from emperor.base.utils import ConfigBase, Module
from emperor.neuron.core.base import NeuronClusterModuleBase
from emperor.neuron.core.config import NeuronClusterConfig, TerminalConfig
from emperor.neuron.core.plasticity import NeuronClusterPlasticityMixin
from emperor.neuron.core._validator import NeuronClusterValidator
from emperor.neuron.core.state import (
    NeuronClusterRouteState,
    NeuronClusterTrace,
    NeuronClusterTraceStep,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase


class NeuronCluster(NeuronClusterModuleBase, NeuronClusterPlasticityMixin):
    """3D neuron grid with sampler-routed recurrent signal propagation.

    The auxiliary loss returned by ``forward`` is an unnormalized sum over
    every routed neuron call and recurrent step, so its scale grows with
    route depth and fan-out. Weight it accordingly when adding it to a
    training objective.
    """

    def __init__(
        self,
        cfg: NeuronClusterConfig,
        overrides: NeuronClusterConfig | None = None,
    ):
        super().__init__()
        self.cfg: NeuronClusterConfig = self._override_config(cfg, overrides)
        NeuronClusterValidator.validate(self)

        self.x_axis_total_neurons: int = self.cfg.x_axis_total_neurons
        self.y_axis_total_neurons: int = self.cfg.y_axis_total_neurons
        self.z_axis_total_neurons: int = self.cfg.z_axis_total_neurons
        self.initial_x_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_x_axis_total_neurons,
            self.x_axis_total_neurons,
        )
        self.initial_y_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_y_axis_total_neurons,
            self.y_axis_total_neurons,
        )
        self.initial_z_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_z_axis_total_neurons,
            self.z_axis_total_neurons,
        )
        self.initial_x_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_x_axis_total_neurons,
            self.x_axis_total_neurons,
        )
        self.initial_y_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_y_axis_total_neurons,
            self.y_axis_total_neurons,
        )
        self.initial_z_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_z_axis_total_neurons,
            self.z_axis_total_neurons,
        )
        self.max_steps: int = self.cfg.max_steps
        self.growth_threshold: int | None = self.cfg.growth_threshold
        self.escape_driven_growth_flag: bool = bool(
            self.cfg.escape_driven_growth_flag
        )
        self.mitosis_initialization_flag: bool = bool(
            self.cfg.mitosis_initialization_flag
        )
        self.halting_config = self.cfg.halting_config
        self.input_dim: int = self.cfg.neuron_config.terminal_config.input_dim

        self.register_buffer(
            "entry_coordinates",
            self.__initialize_entry_coordinates(),
            persistent=False,
        )
        if self.escape_driven_growth_flag:
            self.register_buffer(
                "escape_counts",
                torch.zeros(
                    self.x_axis_total_neurons,
                    self.y_axis_total_neurons,
                    self.z_axis_total_neurons,
                    dtype=torch.long,
                ),
                persistent=True,
            )
        else:
            self.escape_counts = None
        self.entry_sampler_config = self.__resolve_entry_sampler_config()
        self.cluster = self.__initialize_cluster()
        self.entry_sampler = self.__build_entry_sampler()
        self.halting_model = self.__build_halting_model()
        self.register_load_state_dict_pre_hook(
            self._reconcile_cluster_with_state_dict
        )

    def __resolve_initial_dimension(
        self,
        configured_value: int | None,
        maximum_value: int,
    ) -> int:
        return maximum_value if configured_value is None else configured_value

    def __resolve_initial_axis_start(
        self,
        initial_value: int,
        maximum_value: int,
    ) -> int:
        return ((maximum_value - initial_value) // 2) + 1

    def __initialize_entry_coordinates(self) -> Tensor:
        x_indices = torch.arange(
            self.initial_x_axis_start,
            self.initial_x_axis_start + self.initial_x_axis_total_neurons,
            dtype=torch.long,
        )
        y_indices = torch.arange(
            self.initial_y_axis_start,
            self.initial_y_axis_start + self.initial_y_axis_total_neurons,
            dtype=torch.long,
        )
        z_indices = torch.tensor([self.initial_z_axis_start], dtype=torch.long)
        return torch.cartesian_prod(x_indices, y_indices, z_indices)

    def __resolve_entry_sampler_config(self):
        if self.cfg.entry_sampler_config is not None:
            return copy.deepcopy(self.cfg.entry_sampler_config)

        entry_count = int(self.entry_coordinates.shape[0])
        sampler_config = copy.deepcopy(
            self.cfg.neuron_config.terminal_config.sampler_config
        )
        sampler_config.num_experts = entry_count
        sampler_config.top_k = min(sampler_config.top_k, entry_count)
        sampler_config.num_topk_samples = min(
            sampler_config.num_topk_samples,
            sampler_config.top_k,
        )
        if sampler_config.top_k == 1:
            sampler_config.normalize_probabilities_flag = False
        if sampler_config.router_config is not None:
            sampler_config.router_config.num_experts = entry_count
        return sampler_config

    def __build_entry_sampler(self):
        if self.entry_sampler_config.router_config is None:
            return self.entry_sampler_config.build()
        return self.entry_sampler_config.build_with_router_input_dim(self.input_dim)

    def __build_halting_model(self) -> "HaltingBase | None":
        return self.__build_from_config(self.halting_config, input_dim=self.input_dim)

    def __build_from_config(
        self,
        config: "ConfigBase | None",
        **kwargs,
    ) -> "Module | None":
        if config is None:
            return None
        declared_fields = {field.name for field in fields(config)}
        overrides = type(config)(
            **{
                name: value
                for name, value in kwargs.items()
                if name in declared_fields
            }
        )
        return config.build(overrides=overrides)

    def __initialize_cluster(self) -> ModuleDict:
        cluster = ModuleDict()
        for x_coordinate in range(
            self.initial_x_axis_start,
            self.initial_x_axis_start + self.initial_x_axis_total_neurons,
        ):
            for y_coordinate in range(
                self.initial_y_axis_start,
                self.initial_y_axis_start + self.initial_y_axis_total_neurons,
            ):
                for z_coordinate in range(
                    self.initial_z_axis_start,
                    self.initial_z_axis_start + self.initial_z_axis_total_neurons,
                ):
                    name = self._neuron_name(
                        x_coordinate,
                        y_coordinate,
                        z_coordinate,
                    )
                    self._add_neuron(
                        cluster,
                        name,
                        self._initialize_neuron(
                            x_coordinate,
                            y_coordinate,
                            z_coordinate,
                        ),
                    )
        return cluster

    def _initialize_neuron(self, x: int, y: int, z: int) -> Module:
        neuron_config = copy.deepcopy(self.cfg.neuron_config)
        terminal_config = neuron_config.terminal_config
        terminal_overrides = TerminalConfig(
            x_axis_position=x,
            y_axis_position=y,
            z_axis_position=z,
        )
        neuron_config.terminal_config = self._override_config(
            terminal_config,
            terminal_overrides,
        )
        return self.__move_to_current_context(neuron_config.build())

    def __move_to_current_context(self, module: Module) -> Module:
        device, dtype = self.__current_device_and_dtype()
        if dtype is None:
            return module.to(device=device)
        return module.to(device=device, dtype=dtype)

    def __current_device_and_dtype(self) -> tuple[torch.device, torch.dtype | None]:
        fallback_device = None
        for parameter in self.parameters():
            fallback_device = parameter.device
            if parameter.is_floating_point() or parameter.is_complex():
                return parameter.device, parameter.dtype

        for buffer in self.buffers():
            if fallback_device is None:
                fallback_device = buffer.device
            if buffer.is_floating_point() or buffer.is_complex():
                return buffer.device, buffer.dtype

        return fallback_device or torch.device("cpu"), None

    def forward(
        self,
        input: Tensor,
        return_trace: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, NeuronClusterTrace]:
        NeuronClusterValidator.validate_forward_input(input)
        self.__validate_feature_dimension(input)
        self._neurons_called_this_forward: set[str] = set()

        flat_input = input.reshape(-1, input.shape[-1])
        output, auxiliary_loss, trace = (
            self.__propagate_signal_through_recurrent_routes(
                flat_input,
                tuple(input.shape),
                return_trace,
            )
        )
        output = output.reshape(*input.shape[:-1], output.shape[-1])

        if self.training:
            self._check_neuron_growth()
        if return_trace:
            return output, auxiliary_loss, trace
        return output, auxiliary_loss

    def __validate_feature_dimension(self, input: Tensor) -> None:
        if input.shape[-1] != self.input_dim:
            raise ValueError(
                "NeuronCluster input feature dimension must match "
                "neuron_config.terminal_config.input_dim, received "
                f"input_dim={self.input_dim} and input shape {tuple(input.shape)}."
            )

    def __propagate_signal_through_recurrent_routes(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
    ) -> tuple[Tensor, Tensor, NeuronClusterTrace | None]:
        route_state = self.__run_entry_routes(input, input_shape, return_trace)

        for _ in range(self.max_steps):
            route_mask = self.__current_route_mask(route_state)
            if not bool(route_mask.any().item()):
                break
            route_state = self.__run_recurrent_route_step(route_state, route_mask)

        route_state = self.__maybe_finalize_cluster_halting(route_state)
        return route_state.hidden, route_state.loss, route_state.trace

    def __run_entry_routes(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
    ) -> NeuronClusterRouteState:
        probabilities, selected_coords, entry_loss = self.__route_entry_input(input)
        entry_called_mask = torch.ones(
            input.shape[0],
            dtype=torch.bool,
            device=input.device,
        )
        branch_outputs, valid_mask, escape_mask = self.__run_process_branches(
            input,
            selected_coords,
            entry_called_mask,
        )
        weighted_candidate = self.__weighted_branch_candidate(
            branch_outputs,
            probabilities,
        )
        halting_state = self.__maybe_update_halting_state(
            None,
            input,
            weighted_candidate,
            entry_called_mask,
        )

        chosen_branch_indices = probabilities.argmax(dim=1)
        chosen_valid_mask = self.__gather_branch_mask(valid_mask, chosen_branch_indices)
        batch_indices = torch.arange(input.shape[0], device=input.device)

        hidden = input.clone()
        positions = torch.zeros(input.shape[0], 3, dtype=torch.long, device=input.device)
        active_mask = chosen_valid_mask.clone()
        escaped_mask = ~chosen_valid_mask
        final_mask = escaped_mask.clone()

        if bool(chosen_valid_mask.any().item()):
            valid_batch_indices = batch_indices[chosen_valid_mask]
            valid_branch_indices = chosen_branch_indices[chosen_valid_mask]
            hidden[valid_batch_indices] = branch_outputs[
                valid_batch_indices,
                valid_branch_indices,
            ]
            positions[valid_batch_indices] = selected_coords[
                valid_batch_indices,
                valid_branch_indices,
            ]

        if bool(escaped_mask.any().item()):
            hidden[escaped_mask] = weighted_candidate[escaped_mask]

        trace = None
        if return_trace:
            trace = NeuronClusterTrace(
                input_shape=input_shape,
                entry_coordinates=self.__detach_trace_tensor(
                    self.entry_coordinates.to(device=input.device)
                ),
                entry_probabilities=self.__detach_trace_tensor(probabilities),
                entry_selected_coordinates=self.__detach_trace_tensor(selected_coords),
                entry_valid_mask=self.__detach_trace_tensor(valid_mask),
                entry_escape_mask=self.__detach_trace_tensor(escape_mask),
                entry_chosen_branch_indices=self.__detach_trace_tensor(
                    chosen_branch_indices
                ),
                entry_halt_mask=self.__detach_trace_tensor(
                    self.__halt_mask_tensor(
                        halting_state,
                        input.shape[0],
                        input.device,
                    )
                ),
                entry_active_mask=self.__detach_trace_tensor(active_mask),
            )

        return NeuronClusterRouteState(
            hidden=hidden,
            positions=positions,
            active_mask=active_mask,
            escaped_mask=escaped_mask,
            final_mask=final_mask,
            halting_state=halting_state,
            loss=input.new_zeros(()) + entry_loss,
            trace=trace,
        )

    def __route_entry_input(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, indices, _, auxiliary_loss = (
            self.entry_sampler.sample_probabilities_and_indices(input)
        )
        probabilities = self.__ensure_probability_matrix(probabilities)
        indices = self.__resolve_selected_indices(
            input,
            indices,
            int(self.entry_coordinates.shape[0]),
        )
        indices = self.__ensure_index_matrix(indices)
        entry_coordinates = self.entry_coordinates.to(indices.device)[indices]
        return probabilities, entry_coordinates, auxiliary_loss

    def __current_route_mask(self, route_state: NeuronClusterRouteState) -> Tensor:
        halt_mask = self.__get_halt_mask(route_state.halting_state)
        active_mask = route_state.active_mask & ~route_state.final_mask
        if halt_mask is None:
            return active_mask
        return active_mask & ~halt_mask

    def __run_recurrent_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
    ) -> NeuronClusterRouteState:
        next_hidden = route_state.hidden.clone()
        next_positions = route_state.positions.clone()
        next_active_mask = route_state.active_mask.clone()
        next_escaped_mask = route_state.escaped_mask.clone()
        next_final_mask = route_state.final_mask.clone()
        current_called_mask = torch.zeros_like(route_mask)
        probabilities = None
        selected_coords = None
        loss = route_state.loss

        for neuron_name, batch_indices in self.__group_indices_by_position(
            route_state.positions,
            route_mask,
        ).items():
            index_tensor = self.__index_tensor(batch_indices, route_state.hidden.device)
            if neuron_name not in self.cluster:
                next_active_mask[index_tensor] = False
                next_final_mask[index_tensor] = True
                continue

            route_probabilities, route_coords, neuron_loss = self.__route_neuron(
                self.cluster[neuron_name],
                route_state.hidden.index_select(0, index_tensor),
            )
            probabilities, selected_coords = self.__ensure_route_buffers(
                probabilities,
                selected_coords,
                route_probabilities,
                route_coords,
                route_state.hidden,
            )
            probabilities[index_tensor] = route_probabilities
            selected_coords[index_tensor] = route_coords.to(
                device=route_state.hidden.device,
                dtype=torch.long,
            )
            current_called_mask[index_tensor] = True
            loss = self._accumulate_auxiliary_loss(loss, neuron_loss)

        missing_current_mask = route_mask & ~current_called_mask
        if bool(missing_current_mask.any().item()):
            next_active_mask[missing_current_mask] = False
            next_final_mask[missing_current_mask] = True

        if probabilities is None or selected_coords is None:
            return NeuronClusterRouteState(
                hidden=next_hidden,
                positions=next_positions,
                active_mask=next_active_mask,
                escaped_mask=next_escaped_mask,
                final_mask=next_final_mask,
                halting_state=route_state.halting_state,
                loss=loss,
                trace=route_state.trace,
            )

        branch_outputs, valid_target_mask, escape_mask = self.__run_process_branches(
            route_state.hidden,
            selected_coords,
            current_called_mask,
        )
        weighted_candidate = self.__weighted_branch_candidate(
            branch_outputs,
            probabilities,
        )
        chosen_branch_indices = torch.zeros(
            route_state.hidden.shape[0],
            dtype=torch.long,
            device=route_state.hidden.device,
        )
        if bool(current_called_mask.any().item()):
            chosen_branch_indices[current_called_mask] = probabilities[
                current_called_mask
            ].argmax(dim=1)
        chosen_valid_mask = self.__gather_branch_mask(
            valid_target_mask,
            chosen_branch_indices,
        )
        continuing_mask = current_called_mask & chosen_valid_mask
        escaped_final_mask = current_called_mask & ~chosen_valid_mask

        batch_indices = torch.arange(
            route_state.hidden.shape[0],
            device=route_state.hidden.device,
        )
        if bool(continuing_mask.any().item()):
            continuing_batch_indices = batch_indices[continuing_mask]
            continuing_branch_indices = chosen_branch_indices[continuing_mask]
            next_hidden[continuing_batch_indices] = branch_outputs[
                continuing_batch_indices,
                continuing_branch_indices,
            ]
            next_positions[continuing_batch_indices] = selected_coords[
                continuing_batch_indices,
                continuing_branch_indices,
            ]

        if bool(escaped_final_mask.any().item()):
            next_hidden[escaped_final_mask] = weighted_candidate[escaped_final_mask]
            next_escaped_mask[escaped_final_mask] = True
            next_final_mask[escaped_final_mask] = True

        next_active_mask[current_called_mask] = continuing_mask[current_called_mask]

        halting_state = self.__maybe_update_halting_state(
            route_state.halting_state,
            route_state.hidden,
            weighted_candidate,
            current_called_mask,
        )

        if route_state.trace is not None:
            route_state.trace.steps.append(
                NeuronClusterTraceStep(
                    probabilities=self.__detach_trace_tensor(probabilities),
                    selected_coordinates=self.__detach_trace_tensor(selected_coords),
                    valid_mask=self.__detach_trace_tensor(valid_target_mask),
                    escape_mask=self.__detach_trace_tensor(escape_mask),
                    chosen_branch_indices=self.__detach_trace_tensor(
                        chosen_branch_indices
                    ),
                    halt_mask=self.__detach_trace_tensor(
                        self.__halt_mask_tensor(
                            halting_state,
                            route_state.hidden.shape[0],
                            route_state.hidden.device,
                        )
                    ),
                    active_mask=self.__detach_trace_tensor(next_active_mask),
                )
            )

        return NeuronClusterRouteState(
            hidden=next_hidden,
            positions=next_positions,
            active_mask=next_active_mask,
            escaped_mask=next_escaped_mask,
            final_mask=next_final_mask,
            halting_state=halting_state,
            loss=loss,
            trace=route_state.trace,
        )

    def __route_neuron(self, neuron, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if hasattr(neuron, "route_signal"):
            return neuron.route_signal(hidden)

        _, probabilities, selected_coords, auxiliary_loss = neuron(hidden)
        return probabilities, selected_coords, auxiliary_loss

    def __process_neuron(self, neuron, hidden: Tensor) -> Tensor:
        if hasattr(neuron, "process_signal"):
            return neuron.process_signal(hidden)

        output, _, _, _ = neuron(hidden)
        return output

    def __run_process_branches(
        self,
        source_hidden: Tensor,
        selected_coords: Tensor,
        call_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        branch_outputs = source_hidden.unsqueeze(1).expand(
            -1,
            selected_coords.shape[1],
            -1,
        ).clone()
        target_groups, valid_target_mask, escape_mask = (
            self.__group_valid_process_calls(selected_coords, call_mask)
        )

        for neuron_name, branch_indices in target_groups.items():
            batch_indices = [batch_index for batch_index, _ in branch_indices]
            topk_indices = [topk_index for _, topk_index in branch_indices]
            batch_index_tensor = self.__index_tensor(
                batch_indices,
                source_hidden.device,
            )
            topk_index_tensor = self.__index_tensor(
                topk_indices,
                source_hidden.device,
            )
            output = self.__process_neuron(
                self.cluster[neuron_name],
                source_hidden.index_select(0, batch_index_tensor),
            )
            branch_outputs[batch_index_tensor, topk_index_tensor] = output
            if self.training:
                self._neurons_called_this_forward.add(neuron_name)

        return branch_outputs, valid_target_mask, escape_mask

    def __group_valid_process_calls(
        self,
        selected_coords: Tensor,
        call_mask: Tensor,
    ) -> tuple[dict[str, list[tuple[int, int]]], Tensor, Tensor]:
        target_groups: dict[str, list[tuple[int, int]]] = {}
        valid_target_mask = torch.zeros(
            selected_coords.shape[:2],
            dtype=torch.bool,
            device=selected_coords.device,
        )
        escape_mask = torch.zeros_like(valid_target_mask)
        escaped_missing_positions: list[tuple[int, int, int]] = []
        coordinate_rows = selected_coords.detach().cpu().tolist()
        for batch_index in self.__mask_indices(call_mask):
            for topk_index in range(selected_coords.shape[1]):
                coordinate = self._coordinate_from_row(
                    coordinate_rows[batch_index][topk_index]
                )
                if not self.__is_valid_coordinate(coordinate):
                    escape_mask[batch_index, topk_index] = True
                    if self._is_within_grid_capacity(coordinate):
                        escaped_missing_positions.append(coordinate)
                    continue
                neuron_name = self._neuron_name(*coordinate)
                target_groups.setdefault(neuron_name, []).append(
                    (batch_index, topk_index)
                )
                valid_target_mask[batch_index, topk_index] = True
        self._record_escaped_missing_positions(escaped_missing_positions)
        return target_groups, valid_target_mask, escape_mask

    def __ensure_route_buffers(
        self,
        probabilities: Tensor | None,
        selected_coords: Tensor | None,
        route_probabilities: Tensor,
        route_coords: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if probabilities is not None and selected_coords is not None:
            return probabilities, selected_coords

        batch_size = hidden.shape[0]
        top_k = route_probabilities.shape[1]
        return (
            hidden.new_zeros((batch_size, top_k)),
            torch.zeros(
                batch_size,
                top_k,
                3,
                dtype=torch.long,
                device=hidden.device,
            ),
        )

    def __ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def __resolve_selected_indices(
        self,
        input: Tensor,
        indices: Tensor | None,
        total_routes: int,
    ) -> Tensor:
        if indices is not None:
            return indices
        return torch.arange(
            total_routes,
            device=input.device,
            dtype=torch.long,
        ).expand(input.shape[0], -1)

    def __ensure_index_matrix(self, indices: Tensor) -> Tensor:
        if indices.dim() == 1:
            return indices.unsqueeze(-1)
        return indices

    def __weighted_branch_candidate(
        self,
        branch_outputs: Tensor,
        probabilities: Tensor,
    ) -> Tensor:
        return (branch_outputs * probabilities.unsqueeze(-1)).sum(dim=1)

    def __gather_branch_mask(self, mask: Tensor, branch_indices: Tensor) -> Tensor:
        batch_indices = torch.arange(mask.shape[0], device=mask.device)
        return mask[batch_indices, branch_indices]

    def __maybe_update_halting_state(
        self,
        previous_state: "HaltingStateBase | None",
        current_hidden: Tensor,
        weighted_candidate: Tensor,
        update_mask: Tensor,
    ) -> "HaltingStateBase | None":
        if self.halting_model is None or not bool(update_mask.any().item()):
            return previous_state

        halting_input = current_hidden.clone()
        halting_input[update_mask] = weighted_candidate[update_mask]
        halting_state, _ = self.halting_model.update_halting_state(
            previous_state,
            halting_input,
        )
        return halting_state

    def __maybe_finalize_cluster_halting(
        self,
        route_state: NeuronClusterRouteState,
    ) -> NeuronClusterRouteState:
        if self.halting_model is None or route_state.halting_state is None:
            return route_state

        hidden, ponder_loss = self.halting_model.finalize_weighted_accumulation(
            route_state.halting_state,
            route_state.hidden,
        )
        hidden = torch.where(
            route_state.final_mask.unsqueeze(-1),
            route_state.hidden,
            hidden,
        )
        return NeuronClusterRouteState(
            hidden=hidden,
            positions=route_state.positions,
            active_mask=route_state.active_mask,
            escaped_mask=route_state.escaped_mask,
            final_mask=route_state.final_mask,
            halting_state=route_state.halting_state,
            loss=self._accumulate_auxiliary_loss(
                route_state.loss,
                self.__reduce_ponder_loss(ponder_loss),
            ),
            trace=route_state.trace,
        )

    def __reduce_ponder_loss(self, ponder_loss: Tensor) -> Tensor:
        if ponder_loss.dim() == 0:
            return ponder_loss
        denominator = float(max(ponder_loss.numel(), 1))
        return ponder_loss.sum() / denominator

    def __group_indices_by_position(
        self,
        positions: Tensor,
        mask: Tensor,
    ) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        position_rows = positions.detach().cpu().tolist()
        for batch_index in self.__mask_indices(mask):
            coordinate = self._coordinate_from_row(position_rows[batch_index])
            neuron_name = self._neuron_name(*coordinate)
            groups.setdefault(neuron_name, []).append(batch_index)
        return groups

    def __is_valid_coordinate(self, coordinate: tuple[int, int, int]) -> bool:
        if not self._is_within_grid_capacity(coordinate):
            return False
        return self._neuron_name(*coordinate) in self.cluster

    def __mask_indices(self, mask: Tensor) -> list[int]:
        return torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()

    def __index_tensor(self, indices: list[int], device: torch.device) -> Tensor:
        return torch.tensor(indices, dtype=torch.long, device=device)

    def __get_halt_mask(
        self,
        halting_state: "HaltingStateBase | None",
    ) -> Tensor | None:
        if halting_state is None:
            return None
        halt_mask = getattr(halting_state, "halt_mask", None)
        if halt_mask is None:
            return None
        halt_mask = halt_mask.bool()
        if halt_mask.dim() == 1:
            return halt_mask
        return halt_mask.reshape(halt_mask.shape[0], -1).all(dim=1)

    def __halt_mask_tensor(
        self,
        halting_state: "HaltingStateBase | None",
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        halt_mask = self.__get_halt_mask(halting_state)
        if halt_mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return halt_mask

    def __detach_trace_tensor(self, tensor: Tensor) -> Tensor:
        return tensor.detach().clone()
