from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.config import WeightedResidualConfig
from emperor.layers._composition.residual.pairwise import (
    WeightedPairwiseResidualAbstract,
)

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout


class WeightedResidual(WeightedPairwiseResidualAbstract):
    def __init__(
        self,
        cfg: WeightedResidualConfig,
        overrides: WeightedResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)

    @staticmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        return torch.tensor(0.0)

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
        residual_weight = torch.tanh(raw_mix_coefficient)
        return previous + residual_weight * current
