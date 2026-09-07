from __future__ import annotations

from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.config import AdditiveResidualConfig
from emperor.layers._composition.residual.pairwise import (
    PairwiseResidualAbstract,
)


class AdditiveResidual(PairwiseResidualAbstract):
    def __init__(
        self,
        cfg: AdditiveResidualConfig,
        overrides: AdditiveResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        return current + previous
