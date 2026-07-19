from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.experts._options import DroppedTokenOptions

if TYPE_CHECKING:
    from emperor.experts._config import MixtureOfExpertsConfig


class ExpertCapacityHandler:
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
        self.shuffle_indices = None
        return self.__maybe_apply_capacity_limit(token_indices, batch_size)

    def __maybe_apply_capacity_limit(
        self,
        token_indices: Tensor,
        batch_size: int,
        shuffle_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        no_dropped_indices = torch.tensor(
            [], dtype=token_indices.dtype, device=token_indices.device
        )
        if self.__is_capacity_limiting_disabled():
            return token_indices, no_dropped_indices
        assigned_tokens_count = token_indices.size(0)
        expert_capacity = self.__compute_expert_capacity(batch_size)
        if assigned_tokens_count <= expert_capacity:
            return token_indices, no_dropped_indices
        shuffled_indices = self.__resolve_shuffle_indices(
            assigned_tokens_count, shuffle_indices, token_indices.device
        )
        return self.__maybe_split_by_capacity(
            token_indices, expert_capacity, shuffled_indices
        )

    def __is_capacity_limiting_disabled(self) -> bool:
        return self.capacity_factor == 0

    def __compute_expert_capacity(self, batch_size: int) -> int:
        average_tokens_per_expert = batch_size / self.num_experts
        scaled_expert_capacity = average_tokens_per_expert * self.capacity_factor
        integer_expert_capacity = int(scaled_expert_capacity)
        expert_capacity_at_least_one = max(1, integer_expert_capacity)
        return expert_capacity_at_least_one

    def __resolve_shuffle_indices(
        self,
        assigned_tokens_count: int,
        shuffle_indices: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        if shuffle_indices is None:
            shuffle_indices = torch.randperm(assigned_tokens_count, device=device)
            self.shuffle_indices = shuffle_indices
        return shuffle_indices

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

    def maybe_apply_capacity_limit_routing_positions(
        self,
        routing_positions: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        if self.shuffle_indices is None:
            empty_tensor = torch.tensor(
                [], dtype=routing_positions.dtype, device=routing_positions.device
            )
            return routing_positions, empty_tensor
        return self.__maybe_apply_capacity_limit(
            routing_positions, batch_size, self.shuffle_indices
        )

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
