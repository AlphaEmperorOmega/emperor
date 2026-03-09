import torch

from torch import Tensor
from emperor.experts.utils.enums import DroppedTokenOptions


class _ExpertCapacityHandler:
    def __init__(
        self,
        capacity_factor: float,
        num_experts: int,
        dropped_token_behavior: DroppedTokenOptions = DroppedTokenOptions.ZERO,
    ):
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.dropped_token_behavior = dropped_token_behavior

    def maybe_apply_capacity_limit(
        self,
        input: Tensor,
        batch_size: int,
    ) -> Tensor:
        # ) -> tuple[Tensor, Tensor]:
        if self.capacity_factor == 0:
            return input
        tokens_assigned = input.size(0)
        tokens_per_expert = batch_size / self.num_experts
        expert_capacity = max(1, int(tokens_per_expert * self.capacity_factor))
        if tokens_assigned <= expert_capacity:
            return input
        # BUG: applying random shuffleing does not work
        # see if you can see what happends step by step in both situations
        # and if can be somehow avoided, it is importat that this will work
        # see switch transformer implementation:
        # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
        # shuffled = input[torch.randperm(tokens_assigned, device=input.device)]
        # return shuffled[:expert_capacity]

        expert_tokens = input[:expert_capacity]
        droped_tokens = input[expert_capacity:]
        # return expert_tokens, droped_tokens
        return expert_tokens

    def maybe_scatter_to_full_batch(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
        full_size: int,
        input_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.capacity_factor == 0.0 or indices is None:
            return expert_outputs, probabilities, indices

        all_probabilities = probabilities.flatten()
        probabilities = all_probabilities[indices]
        output_dim = expert_outputs.size(-1)

        indices_expanded = indices.unsqueeze(1).expand(-1, output_dim)
        is_identity = (
            self.dropped_token_behavior == DroppedTokenOptions.IDENTITY
            and input_batch is not None
            and input_batch.size(-1) == output_dim
        )
        if is_identity:
            if input_batch.size(0) != full_size:
                repeat_factor = full_size // input_batch.size(0)
                full_expert_outputs = input_batch.repeat(repeat_factor, 1)
            else:
                full_expert_outputs = input_batch
        else:
            full_expert_outputs = torch.zeros(
                full_size,
                output_dim,
                device=expert_outputs.device,
                dtype=expert_outputs.dtype,
            )
        full_expert_outputs.scatter_(0, indices_expanded, expert_outputs)

        if is_identity:
            full_probs = all_probabilities
        else:
            full_probs = torch.zeros(
                full_size, device=probabilities.device, dtype=probabilities.dtype
            )
        full_probs[indices] = probabilities

        return full_expert_outputs, full_probs, None
