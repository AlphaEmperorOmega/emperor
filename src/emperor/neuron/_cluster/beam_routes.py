import torch
from torch import Tensor

from emperor.neuron._cluster.state import NeuronClusterRouteState


class _NeuronClusterBeamRoutesMixin:
    def _propagate_signal_through_beam_routes(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor, None]:
        batch_size = input.shape[0]
        route_state = self.__run_entry_routes_with_beams(input)

        for _ in range(self.max_steps):
            route_mask = self._current_route_mask(route_state)
            if not bool(route_mask.any().item()):
                break
            route_state = self.__run_beam_route_step(route_state, route_mask)

        route_state = self._maybe_finalize_cluster_halting(route_state)
        merged_output = self.__merge_beams_into_output(route_state, batch_size)
        return merged_output, route_state.loss, None

    def __run_entry_routes_with_beams(
        self,
        input: Tensor,
    ) -> NeuronClusterRouteState:
        batch_size = input.shape[0]
        beam_width = self.beam_width
        probabilities, selected_coords, entry_loss = self._route_entry_input(input)
        entry_called_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=input.device,
        )
        branch_outputs, valid_branch_mask, _ = self._run_process_branches(
            input,
            selected_coords,
            entry_called_mask,
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

        flattened_hidden = slot_hidden.reshape(batch_size * beam_width, -1)
        flattened_positions = slot_positions.reshape(batch_size * beam_width, 3)
        active_mask = selected_valid_mask.reshape(-1)
        escaped_mask = (usable_slot_mask & ~selected_valid_mask).reshape(-1)
        final_mask = ~active_mask
        beam_path_probabilities = slot_probabilities.reshape(-1)

        halting_state = self._maybe_update_halting_state(
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

    def __run_beam_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
    ) -> NeuronClusterRouteState:
        beam_width = self.beam_width
        flattened_beam_count = route_state.hidden.shape[0]
        batch_size = flattened_beam_count // beam_width
        route_device = route_state.hidden.device

        probabilities = None
        selected_coords = None
        accumulated_loss = route_state.loss
        called_neuron_mask = torch.zeros_like(route_mask)
        callable_route_mask = self._callable_route_mask(
            route_state.positions,
            route_mask,
        )
        for neuron_name, beam_indices in self._group_indices_by_position(
            route_state.positions,
            callable_route_mask,
        ).items():
            beam_index_tensor = self._index_tensor(beam_indices, route_device)
            route_probabilities, route_coords, neuron_loss = self._route_neuron(
                self.cluster[neuron_name],
                route_state.hidden.index_select(0, beam_index_tensor),
            )
            probabilities, selected_coords = self._ensure_route_buffers(
                probabilities,
                selected_coords,
                route_probabilities,
                route_coords,
                route_state.hidden,
            )
            probabilities[beam_index_tensor] = route_probabilities
            selected_coords[beam_index_tensor] = route_coords.to(
                device=route_device,
                dtype=torch.long,
            )
            called_neuron_mask[beam_index_tensor] = True
            accumulated_loss = self._accumulate_auxiliary_loss(
                accumulated_loss,
                neuron_loss,
            )

        missing_route_mask = route_mask & ~callable_route_mask
        if probabilities is None or selected_coords is None:
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

        branch_outputs, valid_target_mask, _ = self._run_process_branches(
            route_state.hidden,
            selected_coords,
            called_neuron_mask,
        )
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
        selected_path_probabilities, selected_pool_indices = candidate_pool.topk(
            beam_width,
            dim=1,
        )
        usable_slot_mask = (
            torch.isfinite(selected_path_probabilities)
            & (selected_path_probabilities > 0)
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
        parent_hidden = route_state.hidden.index_select(0, parent_rows)
        parent_positions = route_state.positions.index_select(0, parent_rows)
        next_hidden = torch.where(
            flattened_expansion_mask.unsqueeze(-1),
            branch_outputs[parent_rows, flattened_branch_indices],
            parent_hidden,
        )
        next_hidden = torch.where(
            flattened_usable_mask.unsqueeze(-1),
            next_hidden,
            torch.zeros_like(next_hidden),
        )
        next_positions = torch.where(
            flattened_expansion_mask.unsqueeze(-1),
            selected_coords[parent_rows, flattened_branch_indices],
            parent_positions,
        )
        next_positions = torch.where(
            flattened_usable_mask.unsqueeze(-1),
            next_positions,
            torch.zeros_like(next_positions),
        )

        selected_valid_target_mask = (
            valid_target_mask[parent_rows, flattened_branch_indices]
            & flattened_expansion_mask
        )
        kept_path_mask = flattened_usable_mask & ~flattened_expansion_mask
        parent_active_mask = route_state.active_mask.index_select(0, parent_rows)
        parent_escaped_mask = route_state.escaped_mask.index_select(0, parent_rows)
        parent_final_mask = route_state.final_mask.index_select(0, parent_rows)
        kept_missing_mask = (
            missing_route_mask.index_select(0, parent_rows) & kept_path_mask
        )

        next_active_mask = (
            selected_valid_target_mask
            | (
                kept_path_mask
                & parent_active_mask
                & ~parent_final_mask
                & ~kept_missing_mask
            )
        ) & flattened_usable_mask
        next_escaped_mask = (
            (flattened_expansion_mask & ~selected_valid_target_mask)
            | (kept_path_mask & parent_escaped_mask)
        ) & flattened_usable_mask
        next_final_mask = (
            ~flattened_usable_mask
            | (flattened_expansion_mask & ~selected_valid_target_mask)
            | (kept_path_mask & (parent_final_mask | kept_missing_mask))
        )

        halting_state = self._gather_halting_state_rows(
            route_state.halting_state,
            parent_rows,
        )
        halting_state = self._maybe_update_halting_state(
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
            beam_path_probabilities=flattened_path_probabilities,
        )

    def __top_beam_slots(self, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        slot_count = min(self.beam_width, probabilities.shape[1])
        slot_probabilities, slot_branch_indices = probabilities.topk(slot_count, dim=1)
        usable_slot_mask = (
            torch.isfinite(slot_probabilities) & (slot_probabilities > 0)
        )
        slot_probabilities = torch.where(
            usable_slot_mask,
            slot_probabilities,
            torch.zeros_like(slot_probabilities),
        )
        padding_count = self.beam_width - slot_count
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

    def __merge_beams_into_output(
        self,
        route_state: NeuronClusterRouteState,
        batch_size: int,
    ) -> Tensor:
        beam_hidden = route_state.hidden.reshape(batch_size, self.beam_width, -1)
        beam_path_probabilities = route_state.beam_path_probabilities.reshape(
            batch_size,
            self.beam_width,
        )
        return (
            beam_hidden * beam_path_probabilities.unsqueeze(-1)
        ).sum(dim=1)
