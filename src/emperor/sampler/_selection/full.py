from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.sampler._config import SamplerConfig
from emperor.sampler._selection.base import SamplerBase
from emperor.sampler._selection.validation import SamplerFullValidator

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
        thresholded_probabilities = self.__apply_dynamic_topk_threshold_mask(
            probability_matrix
        )
        return thresholded_probabilities, None

    def _sample_probabilities_log_scores_and_indices(
        self,
        probabilities: Tensor,
        log_probabilities: Tensor,
    ) -> tuple[Tensor, Tensor, None]:
        if self.threshold == 0.0:
            return probabilities, log_probabilities, None

        below_threshold_mask = probabilities < self.threshold
        unnormalized_thresholded_probabilities = torch.where(
            below_threshold_mask,
            0.0,
            probabilities,
        )
        selected_log_probabilities = torch.where(
            below_threshold_mask,
            torch.full_like(log_probabilities, float("-inf")),
            log_probabilities,
        )
        selected_log_probabilities = self._normalize_log_probabilities(
            unnormalized_thresholded_probabilities,
            selected_log_probabilities,
        )
        thresholded_probabilities = self._normalize_probabilities(
            unnormalized_thresholded_probabilities
        )
        return thresholded_probabilities, selected_log_probabilities, None

    def __apply_dynamic_topk_threshold_mask(self, probabilities: Tensor) -> Tensor:
        if self.threshold == 0.0:
            return probabilities

        below_threshold_mask = probabilities < self.threshold
        thresholded_probabilities = torch.where(
            below_threshold_mask, 0.0, probabilities
        )
        normalized_probabilities = self._normalize_probabilities(
            thresholded_probabilities
        )
        return normalized_probabilities
