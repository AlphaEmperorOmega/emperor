from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingStateBase


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: HaltingStateBase | None = None
