from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.config import WeightedBlendResidualConfig
from emperor.layers._composition.residual.pairwise import (
    WeightedPairwiseResidualAbstract,
)

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout


class WeightedBlendResidual(WeightedPairwiseResidualAbstract):
    DEFAULT_INITIAL_ALPHA = 0.9

    def __init__(
        self,
        cfg: WeightedBlendResidualConfig,
        overrides: WeightedBlendResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)

    @staticmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        initial_alpha = WeightedBlendResidual.DEFAULT_INITIAL_ALPHA
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        return torch.tensor(initial_logit)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
        row_layout: RowLayout | None = None,
    ) -> Tensor:
        raw_mix_coefficient = self._resolve_raw_mix_coefficient(
            current, previous, row_layout
        )
        current_blend_coefficient = torch.sigmoid(raw_mix_coefficient)
        previous_blend_coefficient = 1.0 - current_blend_coefficient
        current_blend_contribution = current_blend_coefficient * current
        previous_blend_contribution = previous_blend_coefficient * previous
        return current_blend_contribution + previous_blend_contribution
