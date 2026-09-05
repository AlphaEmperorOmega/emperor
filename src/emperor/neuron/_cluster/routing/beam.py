from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._cluster.routing.state import (
    NeuronClusterRouteState,
    _NeuronClusterForwardContext,
)

if TYPE_CHECKING:
    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._cluster.routing.delegate import ClusterRoutingDelegate
    from emperor.neuron._cluster.routing.state import RouteStateDelegate


@dataclass(frozen=True)
class _BeamSelection:
    """Identify retained parent and branch rows for one beam-routing step."""

    parent_rows: Tensor
    branch_indices: Tensor
    expansion_mask: Tensor
    usable_mask: Tensor
    path_probabilities: Tensor


class BeamRoutingDelegate:
    """Expand and retain beam routes using the shared routing lifecycle."""

    def __init__(
        self,
        owner: NeuronCluster,
        state: RouteStateDelegate,
        routing: ClusterRoutingDelegate,
    ) -> None:
        self.__owner = owner
        self.__state = state
        self.__routing = routing

    def propagate(
        self,
        input: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> tuple[Tensor, Tensor, None]:
        batch_size = input.shape[0]
        route_state = self.__run_entry_routes_with_beams(input, forward_context)

        for _ in range(self.__owner.max_steps):
            route_mask = self.__routing.current_route_mask(route_state)
            if not bool(route_mask.any().item()):
                break
            route_state = self.__run_beam_route_step(
                route_state,
                route_mask,
                forward_context,
            )

        route_state = self.__state.maybe_finalize_cluster_halting(route_state)
        merged_output = self.__merge_beams_into_output(route_state, batch_size)
        return merged_output, route_state.loss, None

    def __run_entry_routes_with_beams(
        self,
        input: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> NeuronClusterRouteState:
        batch_size = input.shape[0]
        beam_width = self.__owner.beam_width
        probabilities, selected_coords, entry_loss = self.__routing.route_entry_input(
            input
        )
        entry_called_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=input.device,
        )
        branch_outputs, valid_branch_mask, _ = self.__routing.run_process_branches(
            input,
            selected_coords,
            entry_called_mask,
            forward_context,
        )

        slot_probabilities, slot_branch_indices = self.__top_beam_slots(probabilities)
        usable_slot_mask = slot_probabilities > 0
        batch_indices = torch.arange(batch_size, device=input.device).unsqueeze(1)
        slot_hidden = branch_outputs[batch_indices, slot_branch_indices]
        slot_hidden = torch.where(
            usable_slot_mask.unsqueeze(-1),
            slot_hidden,
            torch.zeros_like(slot_hidden),
        )
        slot_positions = selected_coords[batch_indices, slot_branch_indices]
        slot_positions = torch.where(
            usable_slot_mask.unsqueeze(-1),
            slot_positions,
            torch.zeros_like(slot_positions),
        )
        selected_valid_mask = (
            valid_branch_mask[batch_indices, slot_branch_indices] & usable_slot_mask
        )

        flattened_hidden = slot_hidden.reshape(
            batch_size * beam_width, -1
        )
        flattened_positions = slot_positions.reshape(batch_size * beam_width, 3)
        active_mask = selected_valid_mask.reshape(-1)
        escaped_mask = (usable_slot_mask & ~selected_valid_mask).reshape(-1)
        final_mask = ~active_mask
        beam_path_probabilities = slot_probabilities.reshape(-1)

        halting_state, flattened_hidden = self.__state.maybe_update_halting_state(
            None,
            flattened_hidden,
            flattened_hidden,
            active_mask,
        )

        return NeuronClusterRouteState(
            hidden=flattened_hidden,
            positions=flattened_positions,
            active_mask=active_mask,
            escaped_mask=escaped_mask,
            final_mask=final_mask,
            halting_state=halting_state,
            loss=input.new_zeros(()) + entry_loss,
            trace=None,
            beam_path_probabilities=beam_path_probabilities,
        )

    def __top_beam_slots(self, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        slot_count = min(self.__owner.beam_width, probabilities.shape[1])
        slot_probabilities, slot_branch_indices = probabilities.topk(slot_count, dim=1)
        usable_slot_mask = torch.isfinite(slot_probabilities) & (slot_probabilities > 0)
        slot_probabilities = torch.where(
            usable_slot_mask,
            slot_probabilities,
            torch.zeros_like(slot_probabilities),
        )
        padding_count = self.__owner.beam_width - slot_count
        if padding_count == 0:
            return slot_probabilities, slot_branch_indices
        padding_probabilities = slot_probabilities.new_zeros(
            (slot_probabilities.shape[0], padding_count)
        )
        padding_branch_indices = slot_branch_indices.new_zeros(
            (slot_branch_indices.shape[0], padding_count)
        )
        return (
            torch.cat([slot_probabilities, padding_probabilities], dim=1),
            torch.cat([slot_branch_indices, padding_branch_indices], dim=1),
        )

    def __run_beam_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
        forward_context: _NeuronClusterForwardContext,
    ) -> NeuronClusterRouteState:
        beam_width = self.__owner.beam_width
        flattened_beam_count = route_state.hidden.shape[0]
        batch_size = flattened_beam_count // beam_width
        route_device = route_state.hidden.device

        collection = self.__routing.collect_routes(route_state, route_mask)
        if collection.probabilities is None or collection.coordinates is None:
            return self.__finalize_missing_beam_routes(
                route_state,
                collection.missing_mask,
                collection.loss,
            )

        branch_outputs, valid_target_mask, _ = self.__routing.run_process_branches(
            route_state.hidden,
            collection.coordinates,
            collection.called_mask,
            forward_context,
        )
        selection = self.__select_next_beams(
            route_state,
            collection.probabilities,
            collection.called_mask,
            batch_size,
            beam_width,
            route_device,
        )
        return self.__advance_selected_beams(
            route_state,
            collection.coordinates,
            branch_outputs,
            valid_target_mask,
            collection.missing_mask,
            collection.loss,
            selection,
        )

    def __finalize_missing_beam_routes(
        self,
        route_state: NeuronClusterRouteState,
        missing_route_mask: Tensor,
        accumulated_loss: Tensor,
    ) -> NeuronClusterRouteState:
        return NeuronClusterRouteState(
            hidden=route_state.hidden,
            positions=route_state.positions,
            active_mask=route_state.active_mask & ~missing_route_mask,
            escaped_mask=route_state.escaped_mask,
            final_mask=route_state.final_mask | missing_route_mask,
            halting_state=route_state.halting_state,
            loss=accumulated_loss,
            trace=None,
            beam_path_probabilities=route_state.beam_path_probabilities,
        )

    def __select_next_beams(
        self,
        route_state: NeuronClusterRouteState,
        probabilities: Tensor,
        called_neuron_mask: Tensor,
        batch_size: int,
        beam_width: int,
        route_device: torch.device,
    ) -> _BeamSelection:
        candidate_pool, top_k, expansion_candidate_count = (
            self.__build_beam_candidate_pool(
                route_state,
                probabilities,
                called_neuron_mask,
                batch_size,
                beam_width,
            )
        )
        selected_path_probabilities, selected_pool_indices = candidate_pool.topk(
            beam_width,
            dim=1,
        )
        usable_slot_mask = torch.isfinite(selected_path_probabilities) & (
            selected_path_probabilities > 0
        )
        selected_path_probabilities = torch.where(
            usable_slot_mask,
            selected_path_probabilities,
            torch.zeros_like(selected_path_probabilities),
        )

        expansion_candidate_mask = (
            selected_pool_indices < expansion_candidate_count
        ) & usable_slot_mask
        parent_beam_indices = torch.where(
            selected_pool_indices < expansion_candidate_count,
            selected_pool_indices // top_k,
            selected_pool_indices - expansion_candidate_count,
        )
        branch_indices = torch.where(
            selected_pool_indices < expansion_candidate_count,
            selected_pool_indices % top_k,
            torch.zeros_like(selected_pool_indices),
        )
        sample_beam_offsets = (
            torch.arange(batch_size, device=route_device) * beam_width
        ).unsqueeze(1)
        parent_rows = (sample_beam_offsets + parent_beam_indices).reshape(-1)

        flattened_expansion_mask = expansion_candidate_mask.reshape(-1)
        flattened_usable_mask = usable_slot_mask.reshape(-1)
        flattened_branch_indices = branch_indices.reshape(-1)
        flattened_path_probabilities = selected_path_probabilities.reshape(-1)
        return _BeamSelection(
            parent_rows=parent_rows,
            branch_indices=flattened_branch_indices,
            expansion_mask=flattened_expansion_mask,
            usable_mask=flattened_usable_mask,
            path_probabilities=flattened_path_probabilities,
        )

    def __build_beam_candidate_pool(
        self,
        route_state: NeuronClusterRouteState,
        probabilities: Tensor,
        called_neuron_mask: Tensor,
        batch_size: int,
        beam_width: int,
    ) -> tuple[Tensor, int, int]:
        parent_path_probabilities = route_state.beam_path_probabilities
        positive_branch_mask = (
            called_neuron_mask.unsqueeze(1)
            & torch.isfinite(probabilities)
            & (probabilities > 0)
        )
        expansion_path_probabilities = (
            parent_path_probabilities.unsqueeze(1) * probabilities
        )
        expansion_path_probabilities = torch.where(
            positive_branch_mask
            & torch.isfinite(expansion_path_probabilities)
            & (expansion_path_probabilities > 0),
            expansion_path_probabilities,
            torch.zeros_like(expansion_path_probabilities),
        )

        keep_path_probabilities = torch.where(
            called_neuron_mask,
            torch.zeros_like(parent_path_probabilities),
            parent_path_probabilities,
        )
        top_k = probabilities.shape[1]
        expansion_candidate_count = beam_width * top_k
        candidate_pool = torch.cat(
            [
                expansion_path_probabilities.reshape(
                    batch_size,
                    expansion_candidate_count,
                ),
                keep_path_probabilities.reshape(batch_size, beam_width),
            ],
            dim=1,
        )
        return candidate_pool, top_k, expansion_candidate_count

    def __advance_selected_beams(
        self,
        route_state: NeuronClusterRouteState,
        selected_coords: Tensor,
        branch_outputs: Tensor,
        valid_target_mask: Tensor,
        missing_route_mask: Tensor,
        accumulated_loss: Tensor,
        selection: _BeamSelection,
    ) -> NeuronClusterRouteState:
        next_hidden, next_positions = self.__gather_selected_beam_values(
            route_state,
            selected_coords,
            branch_outputs,
            selection,
        )
        (
            next_active_mask,
            next_escaped_mask,
            next_final_mask,
            selected_valid_target_mask,
        ) = self.__selected_beam_lifecycle_masks(
            route_state,
            valid_target_mask,
            missing_route_mask,
            selection,
        )
        halting_state = self.__state.gather_halting_state_rows(
            route_state.halting_state,
            selection.parent_rows,
            selection.usable_mask,
        )
        halting_state, next_hidden = self.__state.maybe_update_halting_state(
            halting_state,
            next_hidden,
            next_hidden,
            selected_valid_target_mask,
        )

        return NeuronClusterRouteState(
            hidden=next_hidden,
            positions=next_positions,
            active_mask=next_active_mask,
            escaped_mask=next_escaped_mask,
            final_mask=next_final_mask,
            halting_state=halting_state,
            loss=accumulated_loss,
            trace=None,
            beam_path_probabilities=selection.path_probabilities,
        )

    def __gather_selected_beam_values(
        self,
        route_state: NeuronClusterRouteState,
        selected_coords: Tensor,
        branch_outputs: Tensor,
        selection: _BeamSelection,
    ) -> tuple[Tensor, Tensor]:
        parent_hidden = route_state.hidden.index_select(0, selection.parent_rows)
        parent_positions = route_state.positions.index_select(0, selection.parent_rows)
        next_hidden = torch.where(
            selection.expansion_mask.unsqueeze(-1),
            branch_outputs[selection.parent_rows, selection.branch_indices],
            parent_hidden,
        )
        next_hidden = torch.where(
            selection.usable_mask.unsqueeze(-1),
            next_hidden,
            torch.zeros_like(next_hidden),
        )
        next_positions = torch.where(
            selection.expansion_mask.unsqueeze(-1),
            selected_coords[selection.parent_rows, selection.branch_indices],
            parent_positions,
        )
        next_positions = torch.where(
            selection.usable_mask.unsqueeze(-1),
            next_positions,
            torch.zeros_like(next_positions),
        )

        return next_hidden, next_positions

    def __selected_beam_lifecycle_masks(
        self,
        route_state: NeuronClusterRouteState,
        valid_target_mask: Tensor,
        missing_route_mask: Tensor,
        selection: _BeamSelection,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        selected_valid_target_mask = (
            valid_target_mask[selection.parent_rows, selection.branch_indices]
            & selection.expansion_mask
            & selection.usable_mask
        )
        kept_path_mask = selection.usable_mask & ~selection.expansion_mask
        parent_active_mask = route_state.active_mask.index_select(
            0, selection.parent_rows
        )
        parent_escaped_mask = route_state.escaped_mask.index_select(
            0, selection.parent_rows
        )
        parent_final_mask = route_state.final_mask.index_select(
            0, selection.parent_rows
        )
        kept_missing_mask = (
            missing_route_mask.index_select(0, selection.parent_rows) & kept_path_mask
        )

        next_active_mask = (
            selected_valid_target_mask
            | (
                kept_path_mask
                & parent_active_mask
                & ~parent_final_mask
                & ~kept_missing_mask
            )
        ) & selection.usable_mask
        next_escaped_mask = (
            (selection.expansion_mask & ~selected_valid_target_mask)
            | (kept_path_mask & parent_escaped_mask)
        ) & selection.usable_mask
        next_final_mask = (
            ~selection.usable_mask
            | (selection.expansion_mask & ~selected_valid_target_mask)
            | (kept_path_mask & (parent_final_mask | kept_missing_mask))
        )

        return (
            next_active_mask,
            next_escaped_mask,
            next_final_mask,
            selected_valid_target_mask,
        )

    def __merge_beams_into_output(
        self,
        route_state: NeuronClusterRouteState,
        batch_size: int,
    ) -> Tensor:
        beam_hidden = route_state.hidden.reshape(
            batch_size, self.__owner.beam_width, -1
        )
        beam_path_probabilities = route_state.beam_path_probabilities.reshape(
            batch_size,
            self.__owner.beam_width,
        )
        return (beam_hidden * beam_path_probabilities.unsqueeze(-1)).sum(dim=1)
