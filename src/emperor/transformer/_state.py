from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers import LayerState

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase


@dataclass
class TransformerDecoderLayerState(LayerState):
    """Context preserved while decoder blocks pass through Emperor controllers."""

    hidden: Tensor
    loss: Tensor | None = None
    halting_state: "HaltingStateBase | None" = None
    target_key_padding_mask: Tensor | None = None
    target_attention_mask: Tensor | None = None
    encoder_output: Tensor | None = None
    encoder_padding_mask: Tensor | None = None
    cross_attention_mask: Tensor | None = None
    controller_state: object | None = None
