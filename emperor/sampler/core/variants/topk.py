from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler.core._validator import SamplerTopkValidator
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.config import SamplerConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerTopk(SamplerBase):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        SamplerTopkValidator.validate(self)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if self.training and (self.num_topk_samples > 0):
            probabilities, indices = self.__sample_deterministic_and_random_topk(
                probabilities
            )
            return probabilities, indices

        probabilities, indices = torch.topk(probabilities, self.top_k)
        return probabilities, indices

    def __sample_deterministic_and_random_topk(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_deterministic = self.top_k - self.num_topk_samples
        _, topk_deterministic_indices = probabilities.topk(num_deterministic, dim=-1)

        masked_probs = probabilities + 1e-6
        batch_indices = torch.arange(
            probabilities.size(0), device=probabilities.device
        ).unsqueeze(dim=1)
        masked_probs[batch_indices, topk_deterministic_indices] = 0
        topk_random_indices = torch.multinomial(masked_probs, self.num_topk_samples)

        final_topk_indices = torch.concat(
            [topk_deterministic_indices, topk_random_indices], dim=-1
        )

        final_topk_probs = torch.gather(probabilities, 1, final_topk_indices)
        return final_topk_probs, final_topk_indices

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        logits = logits.reshape(-1, self.num_experts)
        gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        full_probabilities = full_probabilities.reshape(-1, self.num_experts)
        skip_mask = self.__prepare_loss_skip_mask(skip_mask)

        self.auxiliary_loss_model.update_accumulated_statistics(
            logits, full_probabilities, gates, skip_mask
        )
        return self.auxiliary_loss_model.get_auxiliary_loss_and_clear()

    def __prepare_loss_gates(
        self, sampled_probabilities: Tensor, indices: Tensor
    ) -> Tensor:
        sampled_probabilities = sampled_probabilities.view(-1, self.top_k)
        indices = indices.view(-1, self.top_k)
        input_dim = sampled_probabilities.shape[0]
        gates_buffer = sampled_probabilities.new_zeros(input_dim, self.num_experts)
        gates = gates_buffer.scatter(1, indices, sampled_probabilities)

        return gates

    def __prepare_loss_skip_mask(
        self, skip_mask: Tensor | None = None
    ) -> Tensor | None:
        if skip_mask is not None:
            return skip_mask.reshape(-1, 1)
        return None
