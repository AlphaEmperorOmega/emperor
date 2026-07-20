from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from emperor.neuron._cluster.halting_lifecycle import _NeuronHaltingLifecycle
from emperor.neuron._trace import NeuronClusterTrace

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase


@dataclass
class NeuronClusterRouteState:
    hidden: Tensor
    positions: Tensor
    active_mask: Tensor
    escaped_mask: Tensor
    final_mask: Tensor
    halting_state: Any | None
    loss: Tensor
    trace: NeuronClusterTrace | None = None
    beam_scores: Tensor | None = None


class _NeuronClusterStateMixin:
    def _ensure_route_buffers(
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

    def _ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def _resolve_selected_indices(
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

    def _ensure_index_matrix(self, indices: Tensor) -> Tensor:
        if indices.dim() == 1:
            return indices.unsqueeze(-1)
        return indices

    def _weighted_branch_candidate(
        self,
        branch_outputs: Tensor,
        probabilities: Tensor,
        valid_branch_mask: Tensor,
    ) -> Tensor:
        valid_weights = probabilities * valid_branch_mask.to(probabilities.dtype)
        valid_weight_sums = valid_weights.sum(dim=1, keepdim=True)
        # With no valid branches every row of branch_outputs holds the
        # unprocessed input, so falling back to the original probabilities
        # passes the input through scaled by their sum — exactly the input
        # only when the sampler normalizes probabilities to 1 — while
        # keeping the gradient path through the router.
        renormalized_weights = torch.where(
            valid_weight_sums > 0,
            valid_weights
            / valid_weight_sums.clamp_min(torch.finfo(probabilities.dtype).tiny),
            probabilities,
        )
        return (branch_outputs * renormalized_weights.unsqueeze(-1)).sum(dim=1)

    def _gather_branch_mask(self, mask: Tensor, branch_indices: Tensor) -> Tensor:
        batch_indices = torch.arange(mask.shape[0], device=mask.device)
        return mask[batch_indices, branch_indices]

    def _maybe_update_halting_state(
        self,
        previous_state: "HaltingStateBase | None",
        current_hidden: Tensor,
        weighted_candidate: Tensor,
        update_mask: Tensor,
    ) -> "HaltingStateBase | None":
        return _NeuronHaltingLifecycle.update(
            self.halting_model,
            previous_state,
            current_hidden,
            weighted_candidate,
            update_mask,
        )

    def _maybe_finalize_cluster_halting(
        self,
        route_state: NeuronClusterRouteState,
    ) -> NeuronClusterRouteState:
        if self.halting_model is None or route_state.halting_state is None:
            return route_state

        finalized_hidden, ponder_loss = _NeuronHaltingLifecycle.finalize(
            self.halting_model,
            route_state.halting_state,
            route_state.hidden,
            route_state.beam_scores,
        )
        finalized_hidden = torch.where(
            route_state.final_mask.unsqueeze(-1),
            route_state.hidden,
            finalized_hidden,
        )
        return NeuronClusterRouteState(
            hidden=finalized_hidden,
            positions=route_state.positions,
            active_mask=route_state.active_mask,
            escaped_mask=route_state.escaped_mask,
            final_mask=route_state.final_mask,
            halting_state=route_state.halting_state,
            loss=self._accumulate_auxiliary_loss(
                route_state.loss,
                ponder_loss,
            ),
            trace=route_state.trace,
            beam_scores=route_state.beam_scores,
        )

    def _group_indices_by_position(
        self,
        positions: Tensor,
        mask: Tensor,
    ) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        position_rows = positions.detach().cpu().tolist()
        for batch_index in self._mask_indices(mask):
            coordinate = self._coordinate_from_row(position_rows[batch_index])
            neuron_name = self._neuron_name(*coordinate)
            groups.setdefault(neuron_name, []).append(batch_index)
        return groups

    def _is_valid_coordinate(self, coordinate: tuple[int, int, int]) -> bool:
        if not self._is_within_grid_capacity(coordinate):
            return False
        return self._neuron_name(*coordinate) in self.cluster

    def _mask_indices(self, mask: Tensor) -> list[int]:
        return torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()

    def _index_tensor(self, indices: list[int], device: torch.device) -> Tensor:
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _get_halt_mask(
        self,
        halting_state: "HaltingStateBase | None",
    ) -> Tensor | None:
        return _NeuronHaltingLifecycle.halt_mask(halting_state)

    def _halt_mask_tensor(
        self,
        halting_state: "HaltingStateBase | None",
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        return _NeuronHaltingLifecycle.halt_mask_tensor(
            halting_state,
            batch_size,
            device,
        )

    def _detach_trace_tensor(self, tensor: Tensor) -> Tensor:
        return tensor.detach().clone()

    def _gather_halting_state_rows(
        self,
        halting_state: "HaltingStateBase | None",
        row_indices: Tensor,
    ) -> "HaltingStateBase | None":
        return _NeuronHaltingLifecycle.gather_state_rows(
            halting_state,
            row_indices,
        )
