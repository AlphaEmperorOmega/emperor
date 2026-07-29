from dataclasses import dataclass

from torch import Tensor

ClassifierBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class ClassifierStepOutput:
    total_loss: Tensor
    logits: Tensor
    labels: Tensor
