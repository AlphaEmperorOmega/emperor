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
        expert_capacity,
        shuffled_indices,
    ) -> tuple[Tensor, Tensor]:
        shuffled_input_tokens = token_indices[shuffled_indices]
        expert_tokens = shuffled_input_tokens[:expert_capacity]
        dropped_tokens = shuffled_input_tokens[expert_capacity:]
        return expert_tokens, dropped_tokens

    def maybe_reconstruct_full_batch_from_expert_outputs(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.capacity_factor == 0 or indices is None:
            return expert_outputs, probabilities, indices

        match self.dropped_token_behavior:
            case DroppedTokenOptions.IDENTITY:
                full_expert_outputs, full_probs = (
                    self.__scatter_expert_outputs_keeping_dropped_tokens(
                        expert_outputs, probabilities, indices, input_batch
                    )
                )
            case DroppedTokenOptions.ZEROS:
                full_expert_outputs, full_probs = (
                    self.__scatter_expert_outputs_zeroing_dropped_tokens(
                        expert_outputs, probabilities, indices
                    )
                )

        return full_expert_outputs, full_probs, None

    def __scatter_expert_outputs_keeping_dropped_tokens(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = input_batch.size(0)
        flat_routing_probabilities = probabilities.flatten()
        sampled_routing_probabilities = flat_routing_probabilities[indices]
        total_routing_slots = flat_routing_probabilities.size(0)
        sample_indices_expanded = indices.unsqueeze(1).expand(
            -1, expert_outputs.size(-1)
        )

        full_expert_outputs = input_batch
        if batch_size != total_routing_slots:
            full_expert_outputs = input_batch.repeat(
                total_routing_slots // batch_size, 1
            )
        full_expert_outputs.scatter_(0, sample_indices_expanded, expert_outputs)

        full_probs = flat_routing_probabilities
        full_probs[indices] = sampled_routing_probabilities

        return full_expert_outputs, full_probs

    def __scatter_expert_outputs_zeroing_dropped_tokens(
        self,
        expert_outputs: Tensor,
        probabilities: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        flat_routing_probabilities = probabilities.flatten()
        sampled_routing_probabilities = flat_routing_probabilities[indices]
        total_routing_slots = flat_routing_probabilities.size(0)
        output_dim = expert_outputs.size(-1)
        sample_indices_expanded = indices.unsqueeze(1).expand(-1, output_dim)

        full_expert_outputs = torch.zeros(
            total_routing_slots,
            output_dim,
            device=expert_outputs.device,
            dtype=expert_outputs.dtype,
        )
        full_expert_outputs.scatter_(0, sample_indices_expanded, expert_outputs)

        full_probs = torch.zeros(
            total_routing_slots,
            device=sampled_routing_probabilities.device,
            dtype=sampled_routing_probabilities.dtype,
        )
        full_probs[indices] = sampled_routing_probabilities

        return full_expert_outputs, full_probs
