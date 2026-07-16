from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from ._metrics import MaskedLanguageModelMetricsLogger

if TYPE_CHECKING:
    from emperor.config import ModelConfig


MaskedLanguageModelBatch = (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
)
MaskedLanguageModelStepOutput = tuple[Tensor, Tensor, Tensor, Tensor | None]


class MaskedLanguageModelExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = MaskedLanguageModelMetricsLogger()

    def training_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_training_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def validation_step(
        self, batch: MaskedLanguageModelBatch, batch_idx: int
    ) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_validation_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def test_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_test_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def _model_step(self, batch: MaskedLanguageModelBatch) -> Tensor:
        loss, _, _, _ = self._model_step_outputs(batch)
        return loss

    def _model_step_outputs(
        self, batch: MaskedLanguageModelBatch
    ) -> MaskedLanguageModelStepOutput:
        input_ids, labels, attention_mask, token_type_ids = self._unpack_batch(batch)
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        forward_kwargs = {}
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask.to(self.device)
        if token_type_ids is not None:
            forward_kwargs["token_type_ids"] = token_type_ids.to(self.device)

        output = self(input_ids, **forward_kwargs)
        auxiliary_loss = None
        if isinstance(output, tuple):
            logits, auxiliary_loss = output[0], output[-1]
        else:
            logits = output

        loss = self.loss_fn(logits.transpose(1, 2), labels)
        if auxiliary_loss is not None and auxiliary_loss.item() != 0.0:
            loss = loss + auxiliary_loss
        return loss, logits, labels, auxiliary_loss

    def _unpack_batch(
        self, batch: MaskedLanguageModelBatch
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if len(batch) == 2:
            input_ids, labels = batch
            return input_ids, labels, None, None
        if len(batch) == 3:
            input_ids, labels, attention_mask = batch
            return input_ids, labels, attention_mask, None
        if len(batch) == 4:
            input_ids, labels, attention_mask, token_type_ids = batch
            return input_ids, labels, attention_mask, token_type_ids
        raise ValueError(
            "MaskedLanguageModelExperiment batches must contain "
            "(input_ids, labels), (input_ids, labels, attention_mask), "
            "or (input_ids, labels, attention_mask, token_type_ids)."
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
