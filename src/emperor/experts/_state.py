from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.layers import LayerState

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class MixtureOfExpertsLayerState(LayerState):
    probabilities: Tensor | None = None
    indices: Tensor | None = None
    skip_mask: Tensor | None = None
