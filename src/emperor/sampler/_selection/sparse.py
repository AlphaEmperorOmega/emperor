from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler._config import SamplerConfig
from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.validation import SamplerSparseValidator

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerSparse(SamplerBase):
    VALIDATOR = SamplerSparseValidator

    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def _sample_probabilities_and_indices(
        self, probabilities: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        expert_dimension = 1
        selected_probabilities, selected_expert_indices = torch.max(
            probabilities, dim=expert_dimension
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
        loss_gates = self.__prepare_loss_gates(sampled_probabilities, indices)
        flattened_router_logits = logits.reshape(-1, self.num_experts)
        flattened_full_probabilities = full_probabilities.reshape(-1, self.num_experts)

        self.auxiliary_loss_model.update_accumulated_statistics(
            flattened_router_logits, flattened_full_probabilities, loss_gates
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
