from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ByteContextualEmbeddingState:
    hidden: Tensor
    byte_moe_auxiliary_loss: Tensor
    context_moe_auxiliary_loss: Tensor
    loss: Tensor
