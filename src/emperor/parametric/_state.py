from dataclasses import dataclass

from torch.types import Tensor

from emperor.layers import LayerState


@dataclass
class ParametricLayerState(LayerState):
    skip_mask: Tensor | None = None
