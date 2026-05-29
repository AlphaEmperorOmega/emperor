from torch.types import Tensor
from dataclasses import dataclass

from emperor.base.layer import LayerState


@dataclass
class ParametricLayerState(LayerState):
    skip_mask: Tensor | None = None
