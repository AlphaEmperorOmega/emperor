import torch

from torch import Tensor
from emperor.base.module import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.model import SamplerModel

    SamplerOutput = tuple[Tensor, Tensor | None, Tensor | None, Tensor]


class SamplerUsageTracker(Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("last_expert_usage_counts", torch.zeros(num_experts))
        self.register_buffer("last_expert_usage_mass", torch.zeros(num_experts))
        self.register_buffer(
            "cumulative_expert_usage_counts", torch.zeros(num_experts)
        )
        self.register_buffer("cumulative_expert_usage_mass", torch.zeros(num_experts))

    def record(self, probabilities: Tensor, indices: Tensor | None) -> None:
        usage_counts, usage_mass = self.compute_usage(probabilities, indices)
        self.last_expert_usage_counts.copy_(usage_counts)
        self.last_expert_usage_mass.copy_(usage_mass)
        self.cumulative_expert_usage_counts.add_(usage_counts)
        self.cumulative_expert_usage_mass.add_(usage_mass)

    def record_sampler_output(self, output: "SamplerOutput") -> None:
        probabilities, indices, _, _ = output
        detached_probabilities = probabilities.detach()
        detached_indices = indices.detach() if indices is not None else None
        self.record(detached_probabilities, detached_indices)

    def compute_usage(
        self,
        probabilities: Tensor,
        indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if indices is None:
            flat_probabilities = probabilities.reshape(-1, self.num_experts)
            usage_counts = (flat_probabilities > 0).sum(dim=0).float()
            usage_mass = flat_probabilities.sum(dim=0)
            return usage_counts, usage_mass

        flat_indices = indices.reshape(-1).long()
        flat_probabilities = probabilities.reshape(-1).float()
        usage_counts = torch.bincount(
            flat_indices, minlength=self.num_experts
        ).float()
        usage_mass = torch.zeros(self.num_experts, device=probabilities.device)
        usage_mass.scatter_add_(0, flat_indices, flat_probabilities)
        return usage_counts, usage_mass

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
        existing_tracker = sampler.usage_tracker
        if existing_tracker is not None:
            return existing_tracker
        tracker = SamplerUsageTracker(sampler.num_experts)
        sampler.add_module(self.TRACKER_MODULE_NAME, tracker)
        return tracker

    def detach(self, sampler: "SamplerModel") -> None:
        if self.TRACKER_MODULE_NAME in sampler._modules:
            del sampler._modules[self.TRACKER_MODULE_NAME]
