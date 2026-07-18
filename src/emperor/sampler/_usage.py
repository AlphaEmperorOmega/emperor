from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.sampler._sampler import SamplerModel

    SamplerOutput = tuple[Tensor, Tensor | None, Tensor | None, Tensor]


class SamplerUsageTracker(Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("last_expert_usage_counts", torch.zeros(num_experts))
        self.register_buffer("last_expert_usage_mass", torch.zeros(num_experts))
        self.register_buffer("cumulative_expert_usage_counts", torch.zeros(num_experts))
        self.register_buffer("cumulative_expert_usage_mass", torch.zeros(num_experts))

    def record(self, probabilities: Tensor, indices: Tensor | None) -> None:
        expert_usage_counts, expert_probability_mass = self.compute_usage(
            probabilities,
            indices,
        )
        self.last_expert_usage_counts.copy_(expert_usage_counts)
        self.last_expert_usage_mass.copy_(expert_probability_mass)
        self.cumulative_expert_usage_counts.add_(expert_usage_counts)
        self.cumulative_expert_usage_mass.add_(expert_probability_mass)

    def record_sampler_output(self, output: "SamplerOutput") -> None:
        normalized_selected_probabilities, selected_expert_indices, _, _ = output
        detached_selected_probabilities = normalized_selected_probabilities.detach()
        detached_selected_expert_indices = (
            selected_expert_indices.detach()
            if selected_expert_indices is not None
            else None
        )
        self.record(
            detached_selected_probabilities,
            detached_selected_expert_indices,
        )

    def compute_usage(
        self,
        probabilities: Tensor,
        indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if indices is None:
            flattened_expert_probabilities = probabilities.reshape(
                -1,
                self.num_experts,
            )
            expert_usage_counts = (
                (flattened_expert_probabilities > 0).sum(dim=0).float()
            )
            expert_probability_mass = flattened_expert_probabilities.sum(dim=0)
            return expert_usage_counts, expert_probability_mass

        flattened_selected_expert_indices = indices.reshape(-1).long()
        flattened_selected_probabilities = probabilities.reshape(-1).float()
        expert_usage_counts = torch.bincount(
            flattened_selected_expert_indices,
            minlength=self.num_experts,
        ).float()
        expert_probability_mass = flattened_selected_probabilities.new_zeros(
            self.num_experts
        )
        expert_probability_mass.scatter_add_(
            0,
            flattened_selected_expert_indices,
            flattened_selected_probabilities,
        )
        return expert_usage_counts, expert_probability_mass

    def reset(self) -> None:
        self.last_expert_usage_counts.zero_()
        self.last_expert_usage_mass.zero_()
        self.cumulative_expert_usage_counts.zero_()
        self.cumulative_expert_usage_mass.zero_()


class SamplerUsageTrackerManager:
    TRACKER_MODULE_NAME = "_usage_tracker"

    @staticmethod
    def maybe_record_sampler_output(
        sampler: "SamplerModel",
        output: "SamplerOutput",
    ) -> None:
        usage_tracker = sampler.usage_tracker
        if usage_tracker is None:
            return
        usage_tracker.record_sampler_output(output)

    def attach(self, sampler: "SamplerModel") -> SamplerUsageTracker:
        existing_usage_tracker = sampler.usage_tracker
        if existing_usage_tracker is not None:
            return existing_usage_tracker
        new_usage_tracker = SamplerUsageTracker(sampler.num_experts)
        sampler.add_module(self.TRACKER_MODULE_NAME, new_usage_tracker)
        return new_usage_tracker

    def detach(self, sampler: "SamplerModel") -> None:
        if self.TRACKER_MODULE_NAME in sampler._modules:
            del sampler._modules[self.TRACKER_MODULE_NAME]
