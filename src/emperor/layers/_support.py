from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.memory import MemoryPositionOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.memory import MemoryInterface


class LayerModuleBase(Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory_model: MemoryInterface | None = None

    def _maybe_apply_memory_by_position(
        self,
        hidden: Tensor,
        position: MemoryPositionOptions,
    ) -> Tensor:
        memory_model = self.memory_model
        memory_is_disabled = memory_model is None
        if memory_is_disabled:
            return hidden

        memory_position_matches = memory_model.memory_position_option == position
        if not memory_position_matches:
            return hidden
        return memory_model(hidden)

    def _reduce_auxiliary_loss(self, loss: Tensor) -> Tensor:
        return loss if loss.dim() == 0 else loss.mean()

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return auxiliary_loss if loss is None else loss + auxiliary_loss
