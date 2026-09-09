from __future__ import annotations

import math

import torch
from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.config import WeightedBlendResidualConfig
from emperor.layers._composition.residual.pairwise import (
    WeightedPairwiseResidualAbstract,
)


class WeightedBlendResidual(WeightedPairwiseResidualAbstract):
    def __init__(
        self,
        cfg: WeightedBlendResidualConfig,
        overrides: WeightedBlendResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)

    @staticmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        initial_alpha = 0.9
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        return torch.tensor(initial_logit)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        raw_mix_coefficient = self._resolve_raw_mix_coefficient(current, previous)
        current_coefficient = torch.sigmoid(raw_mix_coefficient)
        previous_coefficient = 1.0 - current_coefficient
        current_contribution = current_coefficient * current
        previous_contribution = previous_coefficient * previous
        return current_contribution + previous_contribution
