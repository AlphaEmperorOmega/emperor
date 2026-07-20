from dataclasses import dataclass

import torch
from torch import Tensor

from emperor.neuron._cluster.routing_numerics import (
    _forward_value_with_surrogate_gradient,
    _row_has_equal_weight_cancellation,
    _row_has_finite_difference_overflow,
    _stable_beam_mixture_with_score_history,
    _stable_weighted_sum,
)
from emperor.neuron._cluster.state import NeuronClusterRouteState


@dataclass(frozen=True)
class _BeamScoreHistory:
    router_score_events: tuple[Tensor, ...]
    selected_score_indices: tuple[Tensor, ...]


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
        (
            probabilities,
            log_probabilities,
            router_scores,
            selected_coords,
            entry_loss,
        ) = self._route_entry_input_with_router_scores(input)
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
        weighted_candidate = self._weighted_branch_candidate(
            branch_outputs,
            probabilities,
            valid_branch_mask,
            log_probabilities=log_probabilities,
            router_scores=router_scores,
        )

        branch_scores = self.__log_branch_scores(
            probabilities,
            valid_branch_mask,
            log_probabilities=log_probabilities,
            router_scores=router_scores,
        )
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

        flattened_hidden = slot_hidden.reshape(batch_size * beam_width, -1)
        flattened_positions = slot_positions.reshape(batch_size * beam_width, 3)
        active_mask = slot_live_mask.reshape(-1)
        escaped_mask = escaped_slot_mask.reshape(-1)
        final_mask = ~active_mask
        beam_scores = slot_scores.reshape(-1)
        top_k = router_scores.shape[1]
        flattened_score_indices = (
            torch.arange(batch_size, device=input.device).unsqueeze(1) * top_k
            + slot_branch_indices
        )
        flattened_score_indices = torch.where(
            slot_live_mask
            & torch.isfinite(router_scores[batch_indices, slot_branch_indices]),
            flattened_score_indices,
            torch.full_like(flattened_score_indices, -1),
        ).reshape(-1)
        beam_score_history = _BeamScoreHistory(
            router_score_events=(router_scores,),
            selected_score_indices=(flattened_score_indices,),
        )

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
            beam_scores=beam_scores,
            beam_score_history=beam_score_history,
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
        log_probabilities = None
        router_scores = None
        selected_coords = None
        accumulated_loss = route_state.loss
        current_called_mask = torch.zeros_like(route_mask)
        callable_route_mask = self._callable_route_mask(
            route_state.positions,
            route_mask,
        )
        for neuron_name, beam_indices in self._group_indices_by_position(
            route_state.positions,
            callable_route_mask,
        ).items():
            beam_index_tensor = self._index_tensor(beam_indices, route_device)
            (
                route_probabilities,
                route_log_probabilities,
                route_router_scores,
                route_coords,
                neuron_loss,
            ) = self._route_neuron_with_router_scores(
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
            log_probabilities = self._ensure_log_probability_buffer(
                log_probabilities,
                route_log_probabilities,
                route_state.hidden,
            )
            router_scores = self._ensure_log_probability_buffer(
                router_scores,
                route_router_scores,
                route_state.hidden,
            )
            probabilities[beam_index_tensor] = route_probabilities
            log_probabilities[beam_index_tensor] = route_log_probabilities
            router_scores[beam_index_tensor] = route_router_scores
            selected_coords[beam_index_tensor] = route_coords.to(
                device=route_device,
                dtype=torch.long,
            )
            current_called_mask[beam_index_tensor] = True
            accumulated_loss = self._accumulate_auxiliary_loss(
                accumulated_loss,
                neuron_loss,
            )

        if probabilities is None or selected_coords is None:
            missing_route_mask = route_mask & ~callable_route_mask
            return NeuronClusterRouteState(
                hidden=route_state.hidden,
                positions=route_state.positions,
                active_mask=route_state.active_mask & ~missing_route_mask,
                escaped_mask=route_state.escaped_mask,
                final_mask=route_state.final_mask | missing_route_mask,
                halting_state=route_state.halting_state,
                loss=accumulated_loss,
                trace=None,
                beam_scores=route_state.beam_scores,
                beam_score_history=route_state.beam_score_history,
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
            log_probabilities=log_probabilities,
            router_scores=router_scores,
        )
        expansion_scores = route_state.beam_scores.unsqueeze(1) + branch_scores

        has_finite_candidate = torch.isfinite(branch_scores).any(dim=1)
        expandable_mask = current_called_mask & has_finite_candidate
        escaping_mask = current_called_mask & ~has_finite_candidate
        missing_route_mask = route_mask & ~callable_route_mask
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

        expansion_candidate_mask = selected_pool_indices < expansion_candidate_count
        parent_beam_indices = torch.where(
            expansion_candidate_mask,
            selected_pool_indices // top_k,
            selected_pool_indices - expansion_candidate_count,
        )
        branch_indices = torch.where(
            expansion_candidate_mask,
            selected_pool_indices % top_k,
            torch.zeros_like(selected_pool_indices),
        )
        sample_beam_offsets = (
            torch.arange(batch_size, device=route_device) * beam_width
        ).unsqueeze(1)
        parent_rows = (sample_beam_offsets + parent_beam_indices).reshape(-1)

        flattened_expansion_mask = expansion_candidate_mask.reshape(-1)
        flattened_branch_indices = branch_indices.reshape(-1)
        flattened_scores = selected_scores.reshape(-1)
        flattened_live_mask = torch.isfinite(flattened_scores)
        prior_score_history = route_state.beam_score_history
        prior_score_indices = (
            ()
            if prior_score_history is None
            else tuple(
                score_indices.index_select(0, parent_rows)
                for score_indices in prior_score_history.selected_score_indices
            )
        )
        current_score_indices = parent_rows * top_k + flattened_branch_indices
        current_score_indices = torch.where(
            flattened_expansion_mask
            & flattened_live_mask
            & torch.isfinite(router_scores[parent_rows, flattened_branch_indices]),
            current_score_indices,
            torch.full_like(current_score_indices, -1),
        )
        beam_score_history = _BeamScoreHistory(
            router_score_events=(
                (
                    ()
                    if prior_score_history is None
                    else prior_score_history.router_score_events
                )
                + (router_scores,)
            ),
            selected_score_indices=prior_score_indices + (current_score_indices,),
        )

        next_hidden = torch.where(
            flattened_expansion_mask.unsqueeze(-1),
            branch_outputs[parent_rows, flattened_branch_indices],
            route_state.hidden.index_select(0, parent_rows),
        )
        next_positions = torch.where(
            flattened_expansion_mask.unsqueeze(-1),
            selected_coords[parent_rows, flattened_branch_indices],
            route_state.positions.index_select(0, parent_rows),
        )

        parent_active_mask = route_state.active_mask.index_select(0, parent_rows)
        parent_escaped_mask = route_state.escaped_mask.index_select(0, parent_rows)
        parent_final_mask = route_state.final_mask.index_select(0, parent_rows)
        kept_escaping_mask = (
            escaping_mask.index_select(
                0,
                parent_rows,
            )
            & ~flattened_expansion_mask
        )
        kept_missing_mask = (
            missing_route_mask.index_select(
                0,
                parent_rows,
            )
            & ~flattened_expansion_mask
        )

        next_active_mask = (
            flattened_expansion_mask
            | (parent_active_mask & ~kept_escaping_mask & ~kept_missing_mask)
        ) & flattened_live_mask
        next_escaped_mask = ~flattened_expansion_mask & (
            parent_escaped_mask | kept_escaping_mask
        )
        next_final_mask = (
            ~flattened_expansion_mask
            & (parent_final_mask | kept_escaping_mask | kept_missing_mask)
        ) | ~flattened_live_mask

        halting_state = self._gather_halting_state_rows(
            route_state.halting_state,
            parent_rows,
        )
        halting_state = self._maybe_update_halting_state(
            halting_state,
            next_hidden,
            next_hidden,
            flattened_expansion_mask & flattened_live_mask,
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
            beam_scores=flattened_scores,
            beam_score_history=beam_score_history,
        )

    def __log_branch_scores(
        self,
        probabilities: Tensor,
        valid_branch_mask: Tensor,
        *,
        log_probabilities: Tensor | None = None,
        router_scores: Tensor | None = None,
    ) -> Tensor:
        positive_probability_mask = probabilities > 0
        safe_probabilities = torch.where(
            positive_probability_mask,
            probabilities,
            torch.ones_like(probabilities),
        )
        forward_log_probabilities = torch.log(
            safe_probabilities.float()
            if probabilities.dtype in (torch.float16, torch.bfloat16)
            else safe_probabilities
        )
        forward_scores = torch.where(
            valid_branch_mask & positive_probability_mask,
            forward_log_probabilities,
            torch.full_like(forward_log_probabilities, float("-inf")),
        )
        if log_probabilities is None and router_scores is None:
            return forward_scores

        stable_log_probabilities = (
            forward_scores
            if log_probabilities is None
            else (
                log_probabilities.float()
                if log_probabilities.dtype in (torch.float16, torch.bfloat16)
                else log_probabilities
            )
        )
        finite_stable_score_mask = torch.isfinite(stable_log_probabilities)
        subnormal_valid_probability_mask = (
            valid_branch_mask
            & finite_stable_score_mask
            & (probabilities.abs() < torch.finfo(probabilities.dtype).tiny)
        )
        stable_forward_score_row_mask = subnormal_valid_probability_mask.any(
            dim=1,
            keepdim=True,
        )
        forward_scores = torch.where(
            stable_forward_score_row_mask
            & valid_branch_mask
            & finite_stable_score_mask,
            stable_log_probabilities.to(forward_scores.dtype),
            forward_scores,
        )
        stable_gradient_scores = (
            stable_log_probabilities
            if router_scores is None
            else (
                router_scores.float()
                if router_scores.dtype in (torch.float16, torch.bfloat16)
                else router_scores
            )
        )
        finite_gradient_score_mask = torch.isfinite(stable_gradient_scores)
        surrogate_scores = torch.where(
            valid_branch_mask & finite_stable_score_mask & finite_gradient_score_mask,
            stable_gradient_scores,
            torch.full_like(stable_gradient_scores, float("-inf")),
        )
        # Forward scores remain normalized log probabilities. Backward follows
        # the sampler's fixed-selection surrogate through accumulated selected
        # raw scores: top-k/beam membership is held fixed, while sampler
        # normalizer and unselected-logit gradient terms are intentionally omitted.
        return _forward_value_with_surrogate_gradient(
            forward_scores,
            surrogate_scores,
        )

    def __top_beam_slots(self, branch_scores: Tensor) -> tuple[Tensor, Tensor]:
        slot_count = min(self.beam_width, branch_scores.shape[1])
        slot_scores, slot_branch_indices = branch_scores.topk(slot_count, dim=1)
        padding_count = self.beam_width - slot_count
        if padding_count == 0:
            return slot_scores, slot_branch_indices
        padding_scores = slot_scores.new_full(
            (slot_scores.shape[0], padding_count),
            float("-inf"),
        )
        padding_branch_indices = slot_branch_indices.new_zeros(
            (slot_branch_indices.shape[0], padding_count)
        )
        return (
            torch.cat([slot_scores, padding_scores], dim=1),
            torch.cat([slot_branch_indices, padding_branch_indices], dim=1),
        )

    def __merge_beams_into_output(
        self,
        route_state: NeuronClusterRouteState,
        batch_size: int,
    ) -> Tensor:
        beam_width = self.beam_width
        beam_scores = route_state.beam_scores.reshape(batch_size, beam_width)
        beam_hidden = route_state.hidden.reshape(batch_size, beam_width, -1)
        finite_score_mask = torch.isfinite(beam_scores)
        # Entry guarantees slot zero a finite score, but guard against a
        # fully dead row so the softmax cannot produce NaN weights.
        no_finite_score_mask = ~finite_score_mask.any(dim=1, keepdim=True)
        slot_zero_mask = torch.zeros_like(finite_score_mask)
        slot_zero_mask[:, 0] = True
        beam_scores = torch.where(
            no_finite_score_mask & slot_zero_mask,
            torch.zeros_like(beam_scores),
            beam_scores,
        )
        mixture_finite_mask = torch.isfinite(beam_scores)
        promoted_beam_scores = (
            beam_scores.float()
            if beam_scores.dtype in (torch.float16, torch.bfloat16)
            else beam_scores
        )
        promoted_beam_hidden = (
            beam_hidden.float()
            if beam_hidden.dtype in (torch.float16, torch.bfloat16)
            else beam_hidden
        )
        # A tiny weight can underflow before multiplication even though its
        # contribution to a large hidden value is representable. Cast only the
        # completed mixture back to the model dtype.
        beam_weights = torch.softmax(promoted_beam_scores, dim=1)
        mixture_output = (
            (promoted_beam_hidden * beam_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(beam_hidden.dtype)
        )
        subnormal_finite_weight_mask = mixture_finite_mask & (
            beam_weights.abs() < torch.finfo(beam_weights.dtype).tiny
        )
        stable_mixture_row_mask = (
            subnormal_finite_weight_mask.any(dim=1, keepdim=True)
            | _row_has_finite_difference_overflow(beam_hidden, mixture_finite_mask)
            | _row_has_equal_weight_cancellation(
                beam_hidden,
                promoted_beam_scores,
                mixture_finite_mask,
            )
        )
        beam_score_history = route_state.beam_score_history
        stable_mixture_output = (
            _stable_weighted_sum(
                beam_hidden,
                beam_scores,
                mixture_finite_mask,
            )
            if beam_score_history is None
            else _stable_beam_mixture_with_score_history(
                beam_hidden,
                beam_scores,
                mixture_finite_mask,
                beam_score_history.router_score_events,
                beam_score_history.selected_score_indices,
            )
        )
        selected_forward_output = torch.where(
            stable_mixture_row_mask,
            stable_mixture_output,
            mixture_output,
        )
        return _forward_value_with_surrogate_gradient(
            selected_forward_output,
            stable_mixture_output,
        )
