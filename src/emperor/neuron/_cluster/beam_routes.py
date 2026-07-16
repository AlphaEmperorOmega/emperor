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
        output = self.__merge_beams_into_output(route_state, batch_size)
        return output, route_state.loss, None

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
        branch_outputs, valid_mask, _ = self._run_process_branches(
            input,
            selected_coords,
            entry_called_mask,
        )
        weighted_candidate = self._weighted_branch_candidate(
            branch_outputs,
            probabilities,
            valid_mask,
        )

        branch_scores = self.__log_branch_scores(probabilities, valid_mask)
        slot_scores, slot_branch_indices = self.__top_beam_slots(branch_scores)
        slot_live_mask = torch.isfinite(slot_scores)

        batch_indices = torch.arange(batch_size, device=input.device).unsqueeze(1)
        slot_hidden = torch.where(
            slot_live_mask.unsqueeze(-1),
            branch_outputs[batch_indices, slot_branch_indices],
            weighted_candidate.unsqueeze(1),
        )
        slot_positions = selected_coords[batch_indices, slot_branch_indices]
        slot_positions = torch.where(
            slot_live_mask.unsqueeze(-1),
            slot_positions,
            torch.zeros_like(slot_positions),
        )

        # A sample whose every entry branch is invalid escapes immediately:
        # slot zero carries the passthrough candidate with a finite score so
        # the final merge can still select it.
        no_live_slot_mask = ~slot_live_mask.any(dim=1, keepdim=True)
        slot_zero_mask = torch.zeros_like(slot_live_mask)
        slot_zero_mask[:, 0] = True
        escaped_slot_mask = no_live_slot_mask & slot_zero_mask
        slot_scores = torch.where(
            escaped_slot_mask,
            torch.zeros_like(slot_scores),
            slot_scores,
        )

        hidden = slot_hidden.reshape(batch_size * beam_width, -1)
        positions = slot_positions.reshape(batch_size * beam_width, 3)
        active_mask = slot_live_mask.reshape(-1)
        escaped_mask = escaped_slot_mask.reshape(-1)
        final_mask = ~active_mask
        beam_scores = slot_scores.reshape(-1)

        halting_state = self._maybe_update_halting_state(
            None,
            hidden,
            hidden,
            active_mask,
        )

        return NeuronClusterRouteState(
            hidden=hidden,
            positions=positions,
            active_mask=active_mask,
            escaped_mask=escaped_mask,
            final_mask=final_mask,
            halting_state=halting_state,
            loss=input.new_zeros(()) + entry_loss,
            trace=None,
            beam_scores=beam_scores,
        )

    def __run_beam_route_step(
        self,
        route_state: NeuronClusterRouteState,
        route_mask: Tensor,
    ) -> NeuronClusterRouteState:
        beam_width = self.beam_width
        flat_size = route_state.hidden.shape[0]
        batch_size = flat_size // beam_width
        device = route_state.hidden.device

        probabilities = None
        selected_coords = None
        loss = route_state.loss
        current_called_mask = torch.zeros_like(route_mask)
        for neuron_name, beam_indices in self._group_indices_by_position(
            route_state.positions,
            route_mask,
        ).items():
            if neuron_name not in self.cluster:
                continue
            index_tensor = self._index_tensor(beam_indices, device)
            route_probabilities, route_coords, neuron_loss = self._route_neuron(
                self.cluster[neuron_name],
                route_state.hidden.index_select(0, index_tensor),
            )
            probabilities, selected_coords = self._ensure_route_buffers(
                probabilities,
                selected_coords,
                route_probabilities,
                route_coords,
                route_state.hidden,
            )
            probabilities[index_tensor] = route_probabilities
            selected_coords[index_tensor] = route_coords.to(
                device=device,
                dtype=torch.long,
            )
            current_called_mask[index_tensor] = True
            loss = self._accumulate_auxiliary_loss(loss, neuron_loss)

        if probabilities is None or selected_coords is None:
            return NeuronClusterRouteState(
                hidden=route_state.hidden,
                positions=route_state.positions,
                active_mask=route_state.active_mask & ~route_mask,
                escaped_mask=route_state.escaped_mask,
                final_mask=route_state.final_mask | route_mask,
                halting_state=route_state.halting_state,
                loss=loss,
                trace=None,
                beam_scores=route_state.beam_scores,
            )

        branch_outputs, valid_target_mask, _ = self._run_process_branches(
            route_state.hidden,
            selected_coords,
            current_called_mask,
        )

        callable_branch_mask = valid_target_mask & current_called_mask.unsqueeze(1)
        branch_scores = self.__log_branch_scores(
            probabilities,
            callable_branch_mask,
        )
        expansion_scores = route_state.beam_scores.unsqueeze(1) + branch_scores

        has_valid_candidate = callable_branch_mask.any(dim=1)
        expandable_mask = current_called_mask & has_valid_candidate
        escaping_mask = current_called_mask & ~has_valid_candidate
        missing_mask = route_mask & ~current_called_mask
        # Finished beams compete with expansions by their accumulated score;
        # expandable beams must expand, so their keep entry is masked out.
        keep_scores = torch.where(
            expandable_mask,
            torch.full_like(route_state.beam_scores, float("-inf")),
            route_state.beam_scores,
        )

        top_k = probabilities.shape[1]
        expansion_candidate_count = beam_width * top_k
        candidate_pool = torch.cat(
            [
                expansion_scores.reshape(batch_size, expansion_candidate_count),
                keep_scores.reshape(batch_size, beam_width),
            ],
            dim=1,
        )
        selected_scores, selected_pool_indices = candidate_pool.topk(
            beam_width,
            dim=1,
        )

        is_expansion = selected_pool_indices < expansion_candidate_count
        parent_beam_indices = torch.where(
            is_expansion,
            selected_pool_indices // top_k,
            selected_pool_indices - expansion_candidate_count,
        )
        branch_indices = torch.where(
            is_expansion,
            selected_pool_indices % top_k,
            torch.zeros_like(selected_pool_indices),
        )
        sample_offsets = (
            torch.arange(batch_size, device=device) * beam_width
        ).unsqueeze(1)
        parent_rows = (sample_offsets + parent_beam_indices).reshape(-1)

        flat_is_expansion = is_expansion.reshape(-1)
        flat_branch_indices = branch_indices.reshape(-1)
        flat_scores = selected_scores.reshape(-1)
        flat_live_mask = torch.isfinite(flat_scores)

        next_hidden = torch.where(
            flat_is_expansion.unsqueeze(-1),
            branch_outputs[parent_rows, flat_branch_indices],
            route_state.hidden.index_select(0, parent_rows),
        )
        next_positions = torch.where(
            flat_is_expansion.unsqueeze(-1),
            selected_coords[parent_rows, flat_branch_indices],
            route_state.positions.index_select(0, parent_rows),
        )

        parent_active = route_state.active_mask.index_select(0, parent_rows)
        parent_escaped = route_state.escaped_mask.index_select(0, parent_rows)
        parent_final = route_state.final_mask.index_select(0, parent_rows)
        kept_escaping = escaping_mask.index_select(0, parent_rows) & ~flat_is_expansion
        kept_missing = missing_mask.index_select(0, parent_rows) & ~flat_is_expansion

        next_active = (
            flat_is_expansion | (parent_active & ~kept_escaping & ~kept_missing)
        ) & flat_live_mask
        next_escaped = ~flat_is_expansion & (parent_escaped | kept_escaping)
        next_final = (
            ~flat_is_expansion & (parent_final | kept_escaping | kept_missing)
        ) | ~flat_live_mask

        halting_state = self._gather_halting_state_rows(
            route_state.halting_state,
            parent_rows,
        )
        halting_state = self._maybe_update_halting_state(
            halting_state,
            next_hidden,
            next_hidden,
            flat_is_expansion & flat_live_mask,
        )

        return NeuronClusterRouteState(
            hidden=next_hidden,
            positions=next_positions,
            active_mask=next_active,
            escaped_mask=next_escaped,
            final_mask=next_final,
            halting_state=halting_state,
            loss=loss,
            trace=None,
            beam_scores=flat_scores,
        )

    def __log_branch_scores(
        self,
        probabilities: Tensor,
        valid_branch_mask: Tensor,
    ) -> Tensor:
        log_probabilities = torch.log(
            probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        )
        return torch.where(
            valid_branch_mask,
            log_probabilities,
            torch.full_like(log_probabilities, float("-inf")),
        )

    def __top_beam_slots(self, branch_scores: Tensor) -> tuple[Tensor, Tensor]:
        slot_count = min(self.beam_width, branch_scores.shape[1])
        slot_scores, slot_branch_indices = branch_scores.topk(slot_count, dim=1)
        pad_count = self.beam_width - slot_count
        if pad_count == 0:
            return slot_scores, slot_branch_indices
        pad_scores = slot_scores.new_full(
            (slot_scores.shape[0], pad_count),
            float("-inf"),
        )
        pad_indices = slot_branch_indices.new_zeros(
            (slot_branch_indices.shape[0], pad_count)
        )
        return (
            torch.cat([slot_scores, pad_scores], dim=1),
            torch.cat([slot_branch_indices, pad_indices], dim=1),
        )

    def __merge_beams_into_output(
        self,
        route_state: NeuronClusterRouteState,
        batch_size: int,
    ) -> Tensor:
        beam_width = self.beam_width
        scores = route_state.beam_scores.reshape(batch_size, beam_width)
        hidden = route_state.hidden.reshape(batch_size, beam_width, -1)
        finite_mask = torch.isfinite(scores)
        # Entry guarantees slot zero a finite score, but guard against a
        # fully dead row so the softmax cannot produce NaN weights.
        no_finite_mask = ~finite_mask.any(dim=1, keepdim=True)
        slot_zero_mask = torch.zeros_like(finite_mask)
        slot_zero_mask[:, 0] = True
        scores = torch.where(
            no_finite_mask & slot_zero_mask,
            torch.zeros_like(scores),
            scores,
        )
        weights = torch.softmax(scores, dim=1)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)
