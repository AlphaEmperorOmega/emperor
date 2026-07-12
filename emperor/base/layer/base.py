from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.module import Module

if TYPE_CHECKING:
    from emperor.memory.options import MemoryPositionOptions


class LayerModuleBase(Module):
    def _maybe_apply_memory_by_position(
        self,
        hidden: Tensor,
        position: "MemoryPositionOptions",
    ) -> Tensor:
        memory_model = getattr(self, "memory_model", None)
        if memory_model is None:
            return hidden

        memory_position_option = getattr(memory_model, "memory_position_option", None)
        if memory_position_option is None:
            memory_config = getattr(self, "memory_config", None)
            memory_position_option = getattr(
                memory_config, "memory_position_option", None
            )

        if memory_position_option == position:
            return memory_model(hidden)
        return hidden

    def _reduce_auxiliary_loss(self, loss: Tensor) -> Tensor:
        return loss if loss.dim() == 0 else loss.mean()

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return auxiliary_loss if loss is None else loss + auxiliary_loss
