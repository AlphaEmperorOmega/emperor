from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler._config import SamplerConfig
from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.validation import SamplerTopkValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerTopk(SamplerBase):
    VALIDATOR = SamplerTopkValidator

    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if self.training and (self.num_topk_samples > 0):
            selected_probabilities, selected_expert_indices = (
                self.__sample_deterministic_and_random_topk(probabilities)
            )
            return selected_probabilities, selected_expert_indices

        selected_probabilities, selected_expert_indices = torch.topk(
            probabilities, self.top_k
        )
        return selected_probabilities, selected_expert_indices

    def _sample_probabilities_log_scores_and_indices(
        self,
        probabilities: Tensor,
        log_probabilities: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.training and self.num_topk_samples > 0:
            selected_expert_indices = (
                self.__sample_deterministic_and_random_topk_indices(
                    log_probabilities,
                    probabilities,
                )
            )
            selected_probabilities = torch.gather(
                probabilities,
                1,
                selected_expert_indices,
            )
        else:
            _, selected_expert_indices = torch.topk(log_probabilities, self.top_k)
            selected_probabilities = torch.gather(
                probabilities,
                1,
                selected_expert_indices,
            )
        selected_log_probabilities = torch.gather(
            log_probabilities,
            1,
            selected_expert_indices,
        )
        return (
            selected_probabilities,
            selected_log_probabilities,
            selected_expert_indices,
        )

    def __sample_deterministic_and_random_topk_indices(
        self,
        deterministic_ranking_scores: Tensor,
        random_sampling_probabilities: Tensor,
    ) -> Tensor:
        num_deterministic_samples = self.top_k - self.num_topk_samples
        expert_first_scores = deterministic_ranking_scores.transpose(0, 1)
        _, expert_first_deterministic_indices = expert_first_scores.topk(
            num_deterministic_samples,
            dim=0,
        )
        deterministic_topk_indices = expert_first_deterministic_indices.transpose(0, 1)

        sampling_epsilon = 1e-6
        random_sampling_weights = random_sampling_probabilities + sampling_epsilon
        random_sampling_weights.scatter_(1, deterministic_topk_indices, 0)
        random_topk_indices = torch.multinomial(
            random_sampling_weights,
            self.num_topk_samples,
        )
        return torch.hstack((deterministic_topk_indices, random_topk_indices))

    def __sample_deterministic_and_random_topk(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_deterministic_samples = self.top_k - self.num_topk_samples
        expert_first_probabilities = probabilities.transpose(0, 1)
        expert_first_expert_dimension = 0
        _, expert_first_deterministic_indices = expert_first_probabilities.topk(
            num_deterministic_samples,
            dim=expert_first_expert_dimension,
        )
        deterministic_topk_indices = expert_first_deterministic_indices.transpose(0, 1)

        sampling_epsilon = 1e-6
        random_sampling_weights = probabilities + sampling_epsilon
        expert_dimension = 1
        random_sampling_weights.scatter_(
            expert_dimension, deterministic_topk_indices, 0
        )
        random_topk_indices = torch.multinomial(
            random_sampling_weights, self.num_topk_samples
        )

        selected_expert_indices = torch.hstack(
            (deterministic_topk_indices, random_topk_indices)
        )

        selected_probabilities = torch.gather(
            probabilities, expert_dimension, selected_expert_indices
        )
        return selected_probabilities, selected_expert_indices

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        flattened_router_logits = logits.reshape(-1, self.num_experts)
        loss_gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        flattened_full_probabilities = full_probabilities.reshape(-1, self.num_experts)
        flattened_skip_mask = self.__prepare_loss_skip_mask(skip_mask)

        self.auxiliary_loss_model.update_accumulated_statistics(
            flattened_router_logits,
            flattened_full_probabilities,
            loss_gates,
            flattened_skip_mask,
        )
        auxiliary_loss = self.auxiliary_loss_model.get_auxiliary_loss_and_clear()
        return auxiliary_loss

    def __prepare_loss_gates(
        self, sampled_probabilities: Tensor, indices: Tensor
    ) -> Tensor:
        flattened_sampled_probabilities = sampled_probabilities.view(-1, self.top_k)
        flattened_expert_indices = indices.view(-1, self.top_k)
        num_routing_inputs = flattened_sampled_probabilities.shape[0]
        zero_expert_gates = flattened_sampled_probabilities.new_zeros(
            num_routing_inputs, self.num_experts
        )
        expert_dimension = 1
        expert_gates = zero_expert_gates.scatter(
            expert_dimension,
            flattened_expert_indices,
            flattened_sampled_probabilities,
        )

        return expert_gates

    def __prepare_loss_skip_mask(
        self, skip_mask: Tensor | None = None
    ) -> Tensor | None:
        if skip_mask is not None:
            flattened_skip_mask = skip_mask.reshape(-1, 1)
            return flattened_skip_mask
        return None
