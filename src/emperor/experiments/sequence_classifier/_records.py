from dataclasses import dataclass

from torch import Tensor

SequenceClassifierBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class SequenceClassifierStepOutput:
    total_loss: Tensor
    logits: Tensor
    labels: Tensor
