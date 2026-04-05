from torch.types import Tensor
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.utils.options.base import HaltingStateBase


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: "HaltingStateBase | None" = None
