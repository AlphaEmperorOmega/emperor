import torch

from torch import Tensor


class _ExpertCapacityHandler:
    def __init__(self, capacity_factor: float, num_experts: int):
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def maybe_apply_capacity_limit(
        self,
        input: Tensor,
        batch_size: int,
    ) -> Tensor:
        if self.capacity_factor == 0:
            return input
        tokens_assigned = input.size(0)
        tokens_per_expert = batch_size / self.num_experts
        expert_capacity = max(1, int(tokens_per_expert * self.capacity_factor))
        if tokens_assigned <= expert_capacity:
            return input
        shuffled = input[torch.randperm(tokens_assigned, device=input.device)]
        return shuffled[:expert_capacity]

    def maybe_scatter_to_full_batch(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
        full_size: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.capacity_factor == 0.0 or indices is None:
            return expert_outputs, probabilities, indices

        probabilities = probabilities.flatten()[indices]
        output_dim = expert_outputs.size(-1)

        indices_expanded = indices.unsqueeze(1).expand(-1, output_dim)
        full_expert_outputs = torch.zeros(
            full_size,
            output_dim,
            device=expert_outputs.device,
            dtype=expert_outputs.dtype,
        )
        full_expert_outputs.scatter_(0, indices_expanded, expert_outputs)
        full_probs = torch.zeros(
            full_size, device=probabilities.device, dtype=probabilities.dtype
        )
        full_probs[indices] = probabilities

        return full_expert_outputs, full_probs, None
