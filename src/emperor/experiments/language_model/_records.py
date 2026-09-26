from dataclasses import dataclass

from torch import Tensor

LanguageModelBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class LanguageModelStepOutput:
    total_loss: Tensor
    cross_entropy: Tensor
    logits: Tensor
    labels: Tensor
    auxiliary_loss: Tensor


@dataclass(frozen=True)
class HierarchicalLanguageModelStepOutput(LanguageModelStepOutput):
    nll_sum: Tensor
    byte_count: int
