from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler.core._validator import SamplerSparseValidator
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.config import SamplerConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerSparse(SamplerBase):
    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.top_k = 1
        SamplerSparseValidator.validate(self)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        probability, indices = torch.max(probabilities, dim=-1)
        return probability, indices

    def _compute_loss(
        self,
        logits: Tensor,
        full_probabilities: Tensor,
        sampled_probabilities: Tensor,
        indices: Tensor,
        skip_mask: Tensor | None = None,
    ) -> Tensor:
        gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        logits = logits.reshape(-1, self.num_experts)
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
