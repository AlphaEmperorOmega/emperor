from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from emperor.neuron._cluster.routing.halting import _NeuronHaltingLifecycle
from emperor.neuron._trace import NeuronClusterTrace

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.neuron._cluster.model import NeuronCluster
    from emperor.neuron._cluster.topology import ClusterTopologyDelegate


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
    beam_path_probabilities: Tensor | None = None


@dataclass(frozen=True)
class _CollectedRoutes:
    probabilities: Tensor | None
    coordinates: Tensor | None
    called_mask: Tensor
    missing_mask: Tensor
    loss: Tensor


@dataclass
class _NeuronClusterForwardContext:
    """Own transient collaboration state for one cluster forward."""

    called_neuron_names: set[str] = field(default_factory=set)


class RouteStateDelegate:
    """Manage forward-local routing state without owning model tensors."""

    def __init__(
        self,
        owner: NeuronCluster,
        topology: ClusterTopologyDelegate,
    ) -> None:
        self.__owner = owner
        self.__topology = topology

    def ensure_route_buffers(
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


    def ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def resolve_selected_indices(
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

    def ensure_index_matrix(self, indices: Tensor) -> Tensor:
        if indices.dim() == 1:
            return indices.unsqueeze(-1)
        return indices

    def weighted_branch_candidate(
        self,
        branch_outputs: Tensor,
        probabilities: Tensor,
    ) -> Tensor:
        return (branch_outputs * probabilities.unsqueeze(-1)).sum(dim=1)

    def gather_branch_mask(
        self,
        branch_mask: Tensor,
        branch_indices: Tensor,
    ) -> Tensor:
        batch_indices = torch.arange(
            branch_mask.shape[0],
            device=branch_mask.device,
        )
        return branch_mask[batch_indices, branch_indices]

    def maybe_update_halting_state(
        self,
        previous_state: HaltingStateBase | None,
        current_hidden: Tensor,
        weighted_candidate: Tensor,
        update_mask: Tensor,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        return _NeuronHaltingLifecycle.update(
            self.__owner.halting_model,
            previous_state,
            current_hidden,
            weighted_candidate,
            update_mask,
        )

    def maybe_finalize_cluster_halting(
        self,
        route_state: NeuronClusterRouteState,
    ) -> NeuronClusterRouteState:
        if self.__owner.halting_model is None or route_state.halting_state is None:
            return route_state

        finalized_hidden, ponder_loss = _NeuronHaltingLifecycle.finalize(
            self.__owner.halting_model,
            route_state.halting_state,
            route_state.hidden,
            route_state.beam_path_probabilities,
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
            loss=self.accumulate_auxiliary_loss(
                route_state.loss,
                ponder_loss,
            ),
            trace=route_state.trace,
            beam_path_probabilities=route_state.beam_path_probabilities,
        )

    def accumulate_auxiliary_loss(
        self,
        loss: Tensor,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return loss + auxiliary_loss

    def group_indices_by_position(
        self,
        positions: Tensor,
        row_mask: Tensor,
    ) -> dict[str, list[int]]:
        row_indices_by_neuron: dict[str, list[int]] = {}
        position_rows = positions.detach().cpu().tolist()
        for batch_index in self.mask_indices(row_mask):
            route_coordinate = self.__topology.coordinate_from_row(
                position_rows[batch_index]
            )
            neuron_name = self.__topology.neuron_name(*route_coordinate)
            row_indices_by_neuron.setdefault(neuron_name, []).append(batch_index)
        return row_indices_by_neuron

    def mask_indices(self, row_mask: Tensor) -> list[int]:
        return torch.nonzero(row_mask, as_tuple=False).flatten().detach().cpu().tolist()

    def callable_route_mask(self, positions: Tensor, route_mask: Tensor) -> Tensor:
        callable_route_mask = torch.zeros_like(route_mask)
        for neuron_name, row_indices in self.group_indices_by_position(
            positions,
            route_mask,
        ).items():
            if neuron_name not in self.__owner.cluster:
                continue
            callable_route_mask[self.index_tensor(row_indices, route_mask.device)] = (
                True
            )
        return callable_route_mask

    def index_tensor(self, indices: list[int], device: torch.device) -> Tensor:
        return torch.tensor(indices, dtype=torch.long, device=device)

    def get_halt_mask(
        self,
        halting_state: HaltingStateBase | None,
    ) -> Tensor | None:
        return _NeuronHaltingLifecycle.halt_mask(halting_state)

    def halt_mask_tensor(
        self,
        halting_state: HaltingStateBase | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        return _NeuronHaltingLifecycle.halt_mask_tensor(
            halting_state,
            batch_size,
            device,
        )

    def detach_trace_tensor(self, tensor: Tensor) -> Tensor:
        return tensor.detach().clone()

    def gather_halting_state_rows(
        self,
        halting_state: HaltingStateBase | None,
        row_indices: Tensor,
        usable_mask: Tensor | None = None,
    ) -> HaltingStateBase | None:
        return _NeuronHaltingLifecycle.gather_state_rows(
            halting_state,
            row_indices,
            usable_mask,
        )
