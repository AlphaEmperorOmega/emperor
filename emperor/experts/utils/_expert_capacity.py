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
        self.shuffle_indices: Tensor | None = None

    def maybe_apply_capacity_limit_token_indices(
        self,
        input: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        return self.__maybe_apply_capacity_limit(input, batch_size)

    def maybe_apply_capacity_limit_routing_positions(
        self,
        input: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(self.shuffle_indices, Tensor):
            raise RuntimeError(
                "shuffle_indices has not been initialized. maybe_apply_capacity_limit_token_indices must be called first."
            )
        return self.__maybe_apply_capacity_limit(
            input, batch_size, self.shuffle_indices
        )

    def _generate_shuffle_indices(
        self,
        tokens_assigned: int,
        shuffle_indices: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        if shuffle_indices is None:
            shuffle_indices = torch.randperm(tokens_assigned, device=device)
            self.shuffle_indices = shuffle_indices
        return shuffle_indices

    def __maybe_apply_capacity_limit(
        self,
        input: Tensor,
        batch_size: int,
        shuffle_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        empty = torch.tensor([], dtype=input.dtype, device=input.device)
        if self.capacity_factor == 0:
            return input, empty
        tokens_assigned = input.size(0)
        tokens_per_expert = batch_size / self.num_experts
        expert_capacity = max(1, int(tokens_per_expert * self.capacity_factor))
        if tokens_assigned <= expert_capacity:
            return input, empty
        shuffled = input[
            self._generate_shuffle_indices(
                tokens_assigned, shuffle_indices, input.device
            )
        ]
        expert_tokens = shuffled[:expert_capacity]
        dropped_tokens = shuffled[expert_capacity:]
        return expert_tokens, dropped_tokens

    def maybe_scatter_to_full_batch(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.capacity_factor == 0 or indices is None:
            return expert_outputs, probabilities, indices

        batch_size = input_batch.size(0)
        full_size = batch_size * self.top_k
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
