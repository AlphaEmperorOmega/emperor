from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._row_layout import RowLayout


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: HaltingStateBase | None = None
    residual_state: ResidualState | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )
    row_layout: RowLayout | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )
