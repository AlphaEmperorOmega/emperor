from dataclasses import dataclass

from torch.types import Tensor

from emperor.base.layer.state import LayerState


@dataclass
class AttentionLayerState(LayerState):
    key_padding_mask: Tensor | None = None
    attention_mask: Tensor | None = None
