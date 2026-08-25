from __future__ import annotations

from torch import Tensor

from emperor.nn import Module


class RowLayoutAwareModule:
    """Nominal capability for tensor modules accepting ``row_layout=``."""

    pass


class LayerModuleBase(Module):
    def _reduce_auxiliary_loss(self, loss: Tensor) -> Tensor:
        return loss if loss.dim() == 0 else loss.mean()

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        return auxiliary_loss if loss is None else loss + auxiliary_loss
