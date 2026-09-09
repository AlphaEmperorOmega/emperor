from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from emperor.layers._composition.residual.base import ResidualStateLifecycle
from emperor.layers._composition.residual.validation import AttentionResidualValidator
from emperor.layers._composition.residual.variants.attention.state import (
    AttentionResidualState,
)


@dataclass(frozen=True, slots=True)
class AttentionResidualStateLifecycle(ResidualStateLifecycle):
    residual_dim: int
    block_size: int
    validator: type[AttentionResidualValidator]

    def create_state(self, initial_source: Tensor) -> AttentionResidualState:
        self.validator.validate_source(
            initial_source,
            residual_dim=self.residual_dim,
        )
        return AttentionResidualState(
            initial_source,
            block_size=self.block_size,
        )
