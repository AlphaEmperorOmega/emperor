import torch

from torch import Tensor
from emperor.experts.utils.enums import DroppedTokenOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.experts.utils.layers import MixtureOfExpertsConfig


class _ExpertCapacityHandler:
    def __init__(
        self,
        cfg: "MixtureOfExpertsConfig",
    ):
        self.cfg = cfg
        self.capacity_factor = self.cfg.capacity_factor
        self.num_experts = self.cfg.num_experts
        self.top_k = self.cfg.top_k
        self.dropped_token_behavior = self.cfg.dropped_token_behavior
        self.shuffle_indices: Tensor | None = None

    def maybe_apply_capacity_limit_token_indices(
        self,
        token_indices: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        return self.__maybe_apply_capacity_limit(token_indices, batch_size)

    def maybe_apply_capacity_limit_routing_positions(
        self,
        token_indices: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(self.shuffle_indices, Tensor):
            empty_tensor = torch.tensor(
                [], dtype=token_indices.dtype, device=token_indices.device
            )
            return token_indices, empty_tensor
        return self.__maybe_apply_capacity_limit(
            token_indices, batch_size, self.shuffle_indices
        )

    def __maybe_apply_capacity_limit(
        self,
        token_indices: Tensor,
        batch_size: int,
        shuffle_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        empty = torch.tensor([], dtype=token_indices.dtype, device=token_indices.device)
        if self.capacity_factor == 0:
            return token_indices, empty
        assigned_tokens_count = token_indices.size(0)
        expert_capacity = self.__compute_expert_capacity(batch_size)
        if assigned_tokens_count <= expert_capacity:
            return token_indices, empty
        shuffled_indices = self.__resolve_shuffle_indices(
            assigned_tokens_count, shuffle_indices, token_indices.device
        )
        return self.__maybe_split_by_capacity(
            token_indices, expert_capacity, shuffled_indices
        )

    def __compute_expert_capacity(self, batch_size: int) -> int:
        tokens_per_expert = batch_size / self.num_experts
        return max(1, int(tokens_per_expert * self.capacity_factor))

    def __resolve_shuffle_indices(
        self,
        assigned_tokens_count: int,
        indices: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        if indices is None:
            indices = torch.randperm(assigned_tokens_count, device=device)
            self.shuffle_indices = indices
        return indices

    def __maybe_split_by_capacity(
        self,
        token_indices: Tensor,
        expert_capacity: int,
        shuffled_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        shuffled_input_tokens = token_indices[shuffled_indices]
        expert_tokens = shuffled_input_tokens[:expert_capacity]
        dropped_tokens = shuffled_input_tokens[expert_capacity:]
        return expert_tokens, dropped_tokens

    def select_expert_and_dropped_samples(
        self,
        input_batch: Tensor,
        indices: Tensor,
        dropped_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        dropped_tokens = input_batch[dropped_indices]
        if self.dropped_token_behavior == DroppedTokenOptions.ZEROS:
            dropped_tokens = torch.zeros_like(dropped_tokens)
        return input_batch[indices], dropped_tokens

    def maybe_reconstruct_full_batch_from_expert_outputs(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.capacity_factor == 0 or indices is None:
            return expert_outputs, probabilities

        flat_probs = probabilities.flatten()
        total_routing_slots = flat_probs.size(0)
        output_dim = expert_outputs.size(-1)

        sample_indices_expanded = indices.unsqueeze(1).expand(-1, output_dim)
        full_expert_outputs = torch.zeros(
            total_routing_slots,
            output_dim,
            device=expert_outputs.device,
            dtype=expert_outputs.dtype,
        )
        full_expert_outputs.scatter_(0, sample_indices_expanded, expert_outputs)

        sampled_probs = flat_probs[indices]
        full_probs = torch.zeros(
            total_routing_slots,
            device=sampled_probs.device,
            dtype=sampled_probs.dtype,
        )
        full_probs[indices] = sampled_probs

        return full_expert_outputs, full_probs
