from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._cluster.routing.beam import BeamRoutingDelegate
from emperor.neuron._cluster.routing.state import (
    NeuronClusterRouteState,
    RouteStateDelegate,
    _CollectedRoutes,
    _NeuronClusterForwardContext,
)
from emperor.neuron._trace import NeuronClusterTrace, NeuronClusterTraceStep

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
    from emperor.neuron._cluster.topology import ClusterTopologyDelegate


class ClusterRoutingDelegate:
    """Route signals through the owner's live neurons and entry sampler."""

    def __init__(
        self,
        owner: NeuronCluster,
        topology: ClusterTopologyDelegate,
        plasticity: ClusterPlasticityDelegate,
    ) -> None:
        self.__owner = owner
        self.__topology = topology
        self.__plasticity = plasticity
        self.__state = RouteStateDelegate(owner, topology)
        self.__beam_routes = BeamRoutingDelegate(owner, self.__state, self)

    def propagate(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
        forward_context: _NeuronClusterForwardContext,
    ) -> tuple[Tensor, Tensor, NeuronClusterTrace | None]:
        if self.__owner.beam_width > 1:
            return self.__beam_routes.propagate(input, forward_context)

        route_state = self.__run_entry_routes(
            input,
            input_shape,
            return_trace,
            forward_context,
        )

        for _ in range(self.__owner.max_steps):
            route_mask = self.current_route_mask(route_state)
            if not bool(route_mask.any().item()):
                break
            route_state = self.__run_recurrent_route_step(
                route_state,
                route_mask,
                forward_context,
            )

        route_state = self.__state.maybe_finalize_cluster_halting(route_state)
        return route_state.hidden, route_state.loss, route_state.trace

    def __run_entry_routes(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
        forward_context: _NeuronClusterForwardContext,
    ) -> NeuronClusterRouteState:
        probabilities, selected_coords, entry_loss = self.route_entry_input(input)
        entry_called_mask = torch.ones(
            input.shape[0],
            dtype=torch.bool,
            device=input.device,
        )
        branch_outputs, valid_branch_mask, entry_escape_mask = (
            self.run_process_branches(
                input,
                selected_coords,
                entry_called_mask,
                forward_context,
            )
        )
        weighted_candidate = self.__state.weighted_branch_candidate(
            branch_outputs,
            probabilities,
        )
        halting_state, continuation_hidden = self.__state.maybe_update_halting_state(
            None,
            input,
            weighted_candidate,
            valid_branch_mask.any(dim=1),
        )

        chosen_branch_indices, chosen_positions, active_mask = (
            self.__select_entry_routes(
                input, probabilities, selected_coords, valid_branch_mask
            )
        )
        escaped_mask = ~active_mask
        final_mask = escaped_mask.clone()
        continuation_hidden = torch.where(
            active_mask.unsqueeze(-1), continuation_hidden, weighted_candidate
        )
        route_trace = self.__create_entry_trace(
            input,
            input_shape,
            return_trace,
            probabilities,
            selected_coords,
            valid_branch_mask,
            entry_escape_mask,
            chosen_branch_indices,
            halting_state,
            active_mask,
        )

        return NeuronClusterRouteState(
            hidden=continuation_hidden,
            positions=chosen_positions,
            active_mask=active_mask,
            escaped_mask=escaped_mask,
            final_mask=final_mask,
            halting_state=halting_state,
            loss=input.new_zeros(()) + entry_loss,
            trace=route_trace,
        )

    def route_entry_input(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, selected_route_indices, _, auxiliary_loss = (
            self.__owner.entry_sampler.sample_probabilities_and_indices(input)
        )
        probabilities = self.__state.ensure_probability_matrix(probabilities)
        selected_route_indices = self.__state.resolve_selected_indices(
            input,
            selected_route_indices,
            int(self.__owner.entry_coordinates.shape[0]),
        )
        selected_route_indices = self.__state.ensure_index_matrix(
            selected_route_indices
        )
        entry_coordinates = self.__owner.entry_coordinates.to(
            selected_route_indices.device
        )[selected_route_indices]
        return (
            probabilities,
            entry_coordinates,
            auxiliary_loss,
        )

    def run_process_branches(
        self,
        source_hidden: Tensor,
        selected_coords: Tensor,
        call_mask: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> tuple[Tensor, Tensor, Tensor]:
        branch_outputs = (
            source_hidden.unsqueeze(1).expand(-1, selected_coords.shape[1], -1).clone()
        )
        process_calls_by_neuron, valid_target_mask, escape_mask = (
            self.__group_valid_process_calls(selected_coords, call_mask)
        )

        for neuron_name, branch_indices in process_calls_by_neuron.items():
            batch_indices = [batch_index for batch_index, _ in branch_indices]
            topk_indices = [topk_index for _, topk_index in branch_indices]
            batch_index_tensor = self.__state.index_tensor(
                batch_indices,
                source_hidden.device,
            )
            topk_index_tensor = self.__state.index_tensor(
                topk_indices,
                source_hidden.device,
            )
            processed_branch_output = self.process_neuron(
                self.__owner.cluster[neuron_name],
                source_hidden.index_select(0, batch_index_tensor),
            )
            branch_outputs, processed_branch_output = (
                self.__state.promote_floating_values(
                    branch_outputs, processed_branch_output
                )
            )
            branch_outputs[batch_index_tensor, topk_index_tensor] = (
                processed_branch_output
            )
            if self.__owner.training:
                forward_context.called_neuron_names.add(neuron_name)

        return branch_outputs, valid_target_mask, escape_mask

    def __group_valid_process_calls(
        self,
        selected_coords: Tensor,
        call_mask: Tensor,
    ) -> tuple[dict[str, list[tuple[int, int]]], Tensor, Tensor]:
        process_calls_by_neuron: dict[str, list[tuple[int, int]]] = {}
        valid_target_mask = torch.zeros(
            selected_coords.shape[:2],
            dtype=torch.bool,
            device=selected_coords.device,
        )
        escape_mask = torch.zeros_like(valid_target_mask)
        escaped_missing_positions: list[tuple[int, int, int]] = []
        selected_coordinate_rows = selected_coords.detach().cpu().tolist()
        for batch_index in self.__state.mask_indices(call_mask):
            for topk_index in range(selected_coords.shape[1]):
                target_coordinate = self.__topology.coordinate_from_row(
                    selected_coordinate_rows[batch_index][topk_index]
                )
                if not self.__topology.is_valid_coordinate(target_coordinate):
                    escape_mask[batch_index, topk_index] = True
                    if self.__topology.is_within_grid_capacity(target_coordinate):
                        escaped_missing_positions.append(target_coordinate)
                    continue
                neuron_name = self.__topology.neuron_name(*target_coordinate)
                process_calls_by_neuron.setdefault(neuron_name, []).append(
                    (batch_index, topk_index)
                )
                valid_target_mask[batch_index, topk_index] = True
        self.__plasticity.record_escaped_missing_positions(escaped_missing_positions)
        return process_calls_by_neuron, valid_target_mask, escape_mask

    def process_neuron(self, neuron, hidden: Tensor) -> Tensor:
        if hasattr(neuron, "process_signal"):
            processed_output = neuron.process_signal(hidden)
        else:
            processed_output, _, _, _ = neuron(hidden)
        return self.__blend_warming_up_neuron_output(
            neuron,
            hidden,
            processed_output,
        )

    def __blend_warming_up_neuron_output(
        self,
        neuron,
        input: Tensor,
        output: Tensor,
    ) -> Tensor:
        if self.__owner.growth_warmup_steps is None:
            return output
        warmup_remaining = getattr(neuron, "warmup_remaining_steps", None)
        if warmup_remaining is None:
            return output
        remaining_warmup_steps = int(warmup_remaining.item())
        if remaining_warmup_steps <= 0:
            return output
        warmup_step_count = self.__owner.growth_warmup_steps
        processed_output_weight = (
            warmup_step_count - remaining_warmup_steps + 1
        ) / warmup_step_count
        return (
            processed_output_weight * output + (1.0 - processed_output_weight) * input
        )

    def __select_entry_routes(
        self,
        input: Tensor,
        probabilities: Tensor,
        selected_coords: Tensor,
        valid_branch_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        chosen_branch_indices = probabilities.argmax(dim=1)
        chosen_valid_mask = self.__state.gather_branch_mask(
            valid_branch_mask,
            chosen_branch_indices,
        )
        batch_indices = torch.arange(input.shape[0], device=input.device)
        chosen_coordinates = selected_coords[batch_indices, chosen_branch_indices]
        chosen_positions = torch.where(
            chosen_valid_mask.unsqueeze(-1),
            chosen_coordinates,
            torch.zeros_like(chosen_coordinates),
        )
        return chosen_branch_indices, chosen_positions, chosen_valid_mask.clone()

    def __create_entry_trace(
        self,
        input: Tensor,
        input_shape: tuple[int, ...],
        return_trace: bool,
        probabilities: Tensor,
        selected_coords: Tensor,
        valid_branch_mask: Tensor,
        entry_escape_mask: Tensor,
        chosen_branch_indices: Tensor,
        halting_state: HaltingStateBase | None,
        active_mask: Tensor,
    ) -> NeuronClusterTrace | None:
        if not return_trace:
            return None
        return NeuronClusterTrace(
            input_shape=input_shape,
            entry_coordinates=self.__state.detach_trace_tensor(
                self.__owner.entry_coordinates.to(device=input.device)
            ),
            entry_probabilities=self.__state.detach_trace_tensor(probabilities),
            entry_selected_coordinates=self.__state.detach_trace_tensor(
                selected_coords
            ),
            entry_valid_mask=self.__state.detach_trace_tensor(valid_branch_mask),
            entry_escape_mask=self.__state.detach_trace_tensor(entry_escape_mask),
            entry_chosen_branch_indices=self.__state.detach_trace_tensor(
                chosen_branch_indices
            ),
            entry_halt_mask=self.__state.detach_trace_tensor(
                self.__state.halt_mask_tensor(
                    halting_state, input.shape[0], input.device
                )
            ),
            entry_active_mask=self.__state.detach_trace_tensor(active_mask),
        )

    def current_route_mask(self, route_state: NeuronClusterRouteState) -> Tensor:
        halt_mask = self.__state.get_halt_mask(route_state.halting_state)
        active_mask = route_state.active_mask & ~route_state.final_mask
        if halt_mask is None:
            return active_mask
        return active_mask & ~halt_mask

    def __run_recurrent_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> NeuronClusterRouteState:
        next_state = self.__initialize_recurrent_route_state(route_state)
        collection = self.collect_routes(route_state, route_mask)
        next_state.loss = collection.loss
        next_state.active_mask[collection.missing_mask] = False
        next_state.final_mask[collection.missing_mask] = True
        if collection.probabilities is None or collection.coordinates is None:
            next_state.halting_state = route_state.halting_state
            next_state.trace = route_state.trace
            return next_state

        self.__advance_selected_routes(
            route_state,
            next_state,
            collection.probabilities,
            collection.coordinates,
            collection.called_mask,
            forward_context,
        )
        return next_state

    def __initialize_recurrent_route_state(
        self,
        route_state: NeuronClusterRouteState,
    ) -> NeuronClusterRouteState:
        return NeuronClusterRouteState(
            hidden=route_state.hidden.clone(),
            positions=route_state.positions.clone(),
            active_mask=route_state.active_mask.clone(),
            escaped_mask=route_state.escaped_mask.clone(),
            final_mask=route_state.final_mask.clone(),
            halting_state=None,
            loss=route_state.loss,
        )

    def collect_routes(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
    ) -> _CollectedRoutes:
        called_neuron_mask = torch.zeros_like(route_mask)
        probabilities = None
        selected_coords = None
        # Include the prior loss to retain the original floating addition order.
        accumulated_loss = route_state.loss
        callable_route_mask = self.__state.callable_route_mask(
            route_state.positions,
            route_mask,
        )
        for neuron_name, batch_indices in self.__state.group_indices_by_position(
            route_state.positions,
            callable_route_mask,
        ).items():
            batch_index_tensor = self.__state.index_tensor(
                batch_indices, route_state.hidden.device
            )
            route_probabilities, route_coords, neuron_loss = self.route_neuron(
                self.__owner.cluster[neuron_name],
                route_state.hidden.index_select(0, batch_index_tensor),
            )
            probabilities, selected_coords = self.__state.ensure_route_buffers(
                probabilities,
                selected_coords,
                route_probabilities,
                route_coords,
                route_state.hidden,
            )
            probabilities, route_probabilities = self.__state.promote_floating_values(
                probabilities, route_probabilities
            )
            probabilities[batch_index_tensor] = route_probabilities
            selected_coords[batch_index_tensor] = route_coords.to(
                device=route_state.hidden.device,
                dtype=torch.long,
            )
            called_neuron_mask[batch_index_tensor] = True
            accumulated_loss = self.__state.accumulate_auxiliary_loss(
                accumulated_loss, neuron_loss
            )

        missing_route_mask = route_mask & ~callable_route_mask
        return _CollectedRoutes(
            probabilities,
            selected_coords,
            called_neuron_mask,
            missing_route_mask,
            accumulated_loss,
        )

    # Process on arrival and route on departure; a full Neuron forward
    # here would duplicate work and change routing RNG and lifecycle counters.
    def route_neuron(
        self,
        neuron,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if hasattr(neuron, "route_signal"):
            return neuron.route_signal(hidden)

        _, probabilities, selected_coords, auxiliary_loss = neuron(hidden)
        return probabilities, selected_coords, auxiliary_loss

    def __advance_selected_routes(
        self,
        route_state: NeuronClusterRouteState,
        next_state: NeuronClusterRouteState,
        probabilities: Tensor,
        selected_coords: Tensor,
        called_neuron_mask: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> None:
        branch_outputs, valid_target_mask, escape_mask = self.run_process_branches(
            route_state.hidden,
            selected_coords,
            called_neuron_mask,
            forward_context,
        )
        weighted_candidate = self.__state.weighted_branch_candidate(
            branch_outputs, probabilities
        )
        chosen_branch_indices = self.__select_recurrent_branches(
            route_state.hidden, probabilities, called_neuron_mask
        )
        self.__update_selected_route_state(
            next_state,
            selected_coords,
            called_neuron_mask,
            weighted_candidate,
            valid_target_mask,
            chosen_branch_indices,
        )
        next_state.halting_state, continuation_hidden = (
            self.__state.maybe_update_halting_state(
                route_state.halting_state,
                route_state.hidden,
                weighted_candidate,
                valid_target_mask.any(dim=1),
            )
        )
        next_state.hidden = torch.where(
            (called_neuron_mask & next_state.active_mask).unsqueeze(-1),
            continuation_hidden,
            next_state.hidden,
        )
        self.__append_route_trace(
            route_state,
            next_state,
            probabilities,
            selected_coords,
            valid_target_mask,
            escape_mask,
            chosen_branch_indices,
        )
        next_state.trace = route_state.trace

    def __select_recurrent_branches(
        self,
        hidden: Tensor,
        probabilities: Tensor,
        called_neuron_mask: Tensor,
    ) -> Tensor:
        chosen_branch_indices = torch.zeros(
            hidden.shape[0],
            dtype=torch.long,
            device=hidden.device,
        )
        chosen_branch_indices[called_neuron_mask] = probabilities[
            called_neuron_mask
        ].argmax(dim=1)
        return chosen_branch_indices

    def __update_selected_route_state(
        self,
        next_state: NeuronClusterRouteState,
        selected_coords: Tensor,
        called_neuron_mask: Tensor,
        weighted_candidate: Tensor,
        valid_target_mask: Tensor,
        chosen_branch_indices: Tensor,
    ) -> None:
        next_state.hidden, weighted_candidate = self.__state.promote_floating_values(
            next_state.hidden, weighted_candidate
        )
        chosen_valid_mask = self.__state.gather_branch_mask(
            valid_target_mask, chosen_branch_indices
        )
        continuing_route_mask = called_neuron_mask & chosen_valid_mask
        escaped_final_mask = called_neuron_mask & ~chosen_valid_mask
        batch_indices = torch.arange(
            next_state.hidden.shape[0],
            device=next_state.hidden.device,
        )
        if bool(continuing_route_mask.any().item()):
            continuing_batch_indices = batch_indices[continuing_route_mask]
            continuing_branch_indices = chosen_branch_indices[continuing_route_mask]
            next_state.hidden[continuing_batch_indices] = weighted_candidate[
                continuing_batch_indices
            ]
            next_state.positions[continuing_batch_indices] = selected_coords[
                continuing_batch_indices,
                continuing_branch_indices,
            ]

        if bool(escaped_final_mask.any().item()):
            next_state.hidden[escaped_final_mask] = weighted_candidate[
                escaped_final_mask
            ]
            next_state.escaped_mask[escaped_final_mask] = True
            next_state.final_mask[escaped_final_mask] = True
        next_state.active_mask[called_neuron_mask] = continuing_route_mask[
            called_neuron_mask
        ]

    def __append_route_trace(
        self,
        route_state: NeuronClusterRouteState,
        next_state: NeuronClusterRouteState,
        probabilities: Tensor,
        selected_coords: Tensor,
        valid_target_mask: Tensor,
        escape_mask: Tensor,
        chosen_branch_indices: Tensor,
    ) -> None:
        if route_state.trace is None:
            return
        route_state.trace.steps.append(
            NeuronClusterTraceStep(
                probabilities=self.__state.detach_trace_tensor(probabilities),
                selected_coordinates=self.__state.detach_trace_tensor(selected_coords),
                valid_mask=self.__state.detach_trace_tensor(valid_target_mask),
                escape_mask=self.__state.detach_trace_tensor(escape_mask),
                chosen_branch_indices=self.__state.detach_trace_tensor(
                    chosen_branch_indices
                ),
                halt_mask=self.__state.detach_trace_tensor(
                    self.__state.halt_mask_tensor(
                        next_state.halting_state,
                        route_state.hidden.shape[0],
                        route_state.hidden.device,
                    )
                ),
                active_mask=self.__state.detach_trace_tensor(next_state.active_mask),
            )
        )
