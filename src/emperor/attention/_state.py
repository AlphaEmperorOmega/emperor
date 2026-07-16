"""Private attention state implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.layers import LayerState

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class AttentionLayerState(LayerState):
    key_padding_mask: Tensor | None = None
    attention_mask: Tensor | None = None
