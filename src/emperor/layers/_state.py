from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.attention_residual import AttentionResidualState


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: HaltingStateBase | None = None
    residual_state: AttentionResidualState | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )
