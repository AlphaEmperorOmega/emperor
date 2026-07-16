from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from emperor.neuron._trace import NeuronClusterTrace


@dataclass(frozen=True)
class _NeuronObservation:
    trace: "NeuronClusterTrace"
    auxiliary_loss: Tensor


@dataclass(frozen=True)
class _RouteDiagnosticMetrics:
    route_depth: Tensor
    recurrent_steps: float
    escape_fraction: Tensor
    valid_fraction: Tensor
    halted_fraction: Tensor
    active_neuron_count: Tensor
    survival: Tensor


@dataclass(frozen=True)
class _EntryRoutingMetrics:
    mean_entropy: Tensor
    marginal_entropy: Tensor
    coefficient_of_variation: Tensor


class _NeuronDiagnostics:
    @classmethod
    def calculate_route(
        cls,
        trace: "NeuronClusterTrace",
    ) -> _RouteDiagnosticMetrics:
        route_depth = trace.entry_active_mask.detach().float()
        for route_step in trace.steps:
            route_depth = route_depth + route_step.active_mask.detach().float()
        escape_masks = [trace.entry_escape_mask] + [
            route_step.escape_mask for route_step in trace.steps
        ]
        valid_masks = [trace.entry_valid_mask] + [
            route_step.valid_mask for route_step in trace.steps
        ]
        final_halt_mask = (
            trace.steps[-1].halt_mask if trace.steps else trace.entry_halt_mask
        )
        survival_stages = [trace.entry_active_mask.detach().float().mean()]
        survival_stages.extend(
            route_step.active_mask.detach().float().mean() for route_step in trace.steps
        )
        return _RouteDiagnosticMetrics(
            route_depth=route_depth,
            recurrent_steps=float(len(trace.steps)),
            escape_fraction=cls._average_mask_fraction(escape_masks),
            valid_fraction=cls._average_mask_fraction(valid_masks),
            halted_fraction=final_halt_mask.detach().float().mean(),
            active_neuron_count=cls._count_active_neurons(trace),
            survival=torch.stack(survival_stages),
        )

    @classmethod
    def calculate_entry_routing(
        cls,
        trace: "NeuronClusterTrace",
    ) -> _EntryRoutingMetrics | None:
        probabilities = trace.entry_probabilities.detach().float()
        if probabilities.numel() == 0:
            return None
        normalized_probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-9)
        per_sample_entropy = cls._distribution_entropy(
            normalized_probabilities,
            dimension=-1,
        )
        marginal_probabilities = normalized_probabilities.mean(dim=0)
        marginal_probabilities = (
            marginal_probabilities / marginal_probabilities.sum().clamp_min(1e-9)
        )
        return _EntryRoutingMetrics(
            mean_entropy=per_sample_entropy.mean(),
            marginal_entropy=cls._distribution_entropy(
                marginal_probabilities,
                dimension=-1,
            ),
            coefficient_of_variation=(
                marginal_probabilities.std()
                / marginal_probabilities.mean().clamp_min(1e-6)
            ),
        )

    @staticmethod
    def valid_coordinates(
        trace: "NeuronClusterTrace",
    ) -> Iterator[tuple[Tensor, Tensor]]:
        yield trace.entry_selected_coordinates, trace.entry_valid_mask
        for route_step in trace.steps:
            yield route_step.selected_coordinates, route_step.valid_mask

    @staticmethod
    def _average_mask_fraction(masks: list[Tensor]) -> Tensor:
        fractions = [mask.detach().float().mean() for mask in masks if mask.numel() > 0]
        return torch.stack(fractions).mean() if fractions else torch.zeros(())

    @classmethod
    def _count_active_neurons(cls, trace: "NeuronClusterTrace") -> Tensor:
        coordinate_rows = []
        for coordinates, valid_mask in cls.valid_coordinates(trace):
            valid_coordinates = coordinates[valid_mask.bool()]
            if valid_coordinates.numel() > 0:
                coordinate_rows.append(valid_coordinates.reshape(-1, 3))
        if not coordinate_rows:
            return torch.zeros(())
        unique_coordinates = torch.unique(torch.cat(coordinate_rows), dim=0)
        return torch.tensor(float(unique_coordinates.shape[0]))

    @staticmethod
    def _distribution_entropy(
        distribution: Tensor,
        dimension: int,
    ) -> Tensor:
        safe_distribution = distribution.clamp_min(1e-9)
        return -(safe_distribution.log() * distribution).sum(dim=dimension)
