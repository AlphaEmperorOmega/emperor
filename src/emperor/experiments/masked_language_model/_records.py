from dataclasses import dataclass

from torch import Tensor

MaskedLanguageModelBatch = (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
)


@dataclass(frozen=True)
class MaskedLanguageModelStepOutput:
    total_loss: Tensor
    cross_entropy: Tensor
    logits: Tensor
    labels: Tensor
    auxiliary_loss: Tensor | None = None
