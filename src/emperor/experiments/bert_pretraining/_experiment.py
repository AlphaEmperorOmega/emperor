from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from ._metrics import BertPretrainingMetricsLogger
from ._records import BertPretrainingBatch, BertPretrainingStepOutput

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class BertPretrainingExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        self.metrics = BertPretrainingMetricsLogger()

    def training_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_training_step(self.log_dict, step_output)
        return step_output.total_loss

    def validation_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_validation_step(self.log_dict, step_output)
        return step_output.total_loss

    def test_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_test_step(self.log_dict, step_output)
        return step_output.total_loss

    def _model_step(self, batch: BertPretrainingBatch) -> Tensor:
        return self._model_step_outputs(batch).total_loss

    def _model_step_outputs(
        self,
        batch: BertPretrainingBatch,
    ) -> BertPretrainingStepOutput:
        (
            input_ids,
            mlm_labels,
            attention_mask,
            token_type_ids,
            next_sentence_labels,
        ) = self._unpack_batch(batch)
        input_ids = input_ids.to(self.device)
        mlm_labels = mlm_labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        next_sentence_labels = next_sentence_labels.to(self.device)

        mlm_logits, nsp_logits, auxiliary_loss = self(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        mlm_loss = self.mlm_loss_fn(mlm_logits.transpose(1, 2), mlm_labels)
        nsp_loss = self.nsp_loss_fn(nsp_logits, next_sentence_labels)
        total_loss = mlm_loss + nsp_loss
        if auxiliary_loss is not None and bool(
            torch.any(auxiliary_loss.detach() != 0.0).item()
        ):
            total_loss = total_loss + auxiliary_loss
        return BertPretrainingStepOutput(
            total_loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
            mlm_logits=mlm_logits,
            mlm_labels=mlm_labels,
            nsp_logits=nsp_logits,
            next_sentence_labels=next_sentence_labels,
            auxiliary_loss=auxiliary_loss,
        )

    def _unpack_batch(self, batch: BertPretrainingBatch) -> BertPretrainingBatch:
        if len(batch) != 5:
            raise ValueError(
                "BertPretrainingExperiment batches must contain "
                "(input_ids, mlm_labels, attention_mask, token_type_ids, "
                "next_sentence_labels)."
            )
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
