from dataclasses import dataclass

from torch import Tensor

TranslationBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class TranslationStepOutput:
    total_loss: Tensor
    nll: Tensor
    logits: Tensor
    labels: Tensor
    auxiliary_loss: Tensor
