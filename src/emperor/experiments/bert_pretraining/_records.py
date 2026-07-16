from dataclasses import dataclass

from torch import Tensor

BertPretrainingBatch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class BertPretrainingStepOutput:
    total_loss: Tensor
    mlm_loss: Tensor
    nsp_loss: Tensor
    mlm_logits: Tensor
    mlm_labels: Tensor
    nsp_logits: Tensor
    next_sentence_labels: Tensor
    auxiliary_loss: Tensor | None = None
