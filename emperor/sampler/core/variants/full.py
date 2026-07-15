from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler.core._validator import SamplerFullValidator
from emperor.sampler.core.base import SamplerBase
from emperor.sampler.core.config import SamplerConfig

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SamplerFull(SamplerBase):
    VALIDATOR = SamplerFullValidator

    def __init__(
        self,
        cfg: "SamplerConfig | ModelConfig",
        overrides: "SamplerConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def _sample_probabilities_and_indices(
        self, probability_matrix: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        probability_matrix = self.__apply_dynamic_topk_threshold_mask(
            probability_matrix
        )
        return probability_matrix, None

    def __apply_dynamic_topk_threshold_mask(self, probabilities: Tensor) -> Tensor:
        if self.threshold == 0.0:
            return probabilities

        dynamic_topk_mask = probabilities < self.threshold
        masked_probabilities = torch.where(dynamic_topk_mask, 0.0, probabilities)
        normalized_probabilities = self._normalize_probabilities(masked_probabilities)
        return normalized_probabilities
