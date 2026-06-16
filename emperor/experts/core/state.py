from dataclasses import dataclass

from torch import Tensor

from emperor.base.layer.state import LayerState


@dataclass
class MixtureOfExpertsLayerState(LayerState):
    probabilities: Tensor | None = None
    indices: Tensor | None = None
    skip_mask: Tensor | None = None
