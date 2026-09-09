"""Attention residual connection and forward-local state."""

from emperor.layers._composition.residual.variants.attention.core import (
    AttentionResidual,
)
from emperor.layers._composition.residual.variants.attention.state import (
    AttentionResidualState,
)

__all__ = ("AttentionResidual", "AttentionResidualState")
