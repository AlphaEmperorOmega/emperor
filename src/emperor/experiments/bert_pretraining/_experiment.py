from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from emperor.experiments._auxiliary_loss import AuxiliaryLoss

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
        self._auxiliary_loss = AuxiliaryLoss("BERT-pretraining")

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

        mlm_logits, nsp_logits, auxiliary_loss, resolved_auxiliary_loss = (
            self._validate_model_output(
                self(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ),
                mlm_labels,
                next_sentence_labels,
            )
        )
        mlm_loss = self.mlm_loss_fn(mlm_logits.transpose(1, 2), mlm_labels)
        nsp_loss = self.nsp_loss_fn(nsp_logits, next_sentence_labels)
        task_loss = mlm_loss + nsp_loss
        total_loss = task_loss
        if resolved_auxiliary_loss is not None:
            total_loss = task_loss + resolved_auxiliary_loss
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
        input_ids, mlm_labels, attention_mask, token_type_ids, next_sentence_labels = (
            batch
        )
        token_tensors = (input_ids, mlm_labels, attention_mask, token_type_ids)
        if any(
            not isinstance(value, Tensor) or value.ndim != 2
            for value in token_tensors
        ):
            raise ValueError(
                "BERT-pretraining input IDs, MLM labels, attention mask, and "
                "token-type IDs must be rank-2 tensors."
            )
        if any(value.shape != input_ids.shape for value in token_tensors[1:]):
            raise ValueError(
                "BERT-pretraining token inputs and MLM labels must have equal shapes."
            )
        if (
            not isinstance(next_sentence_labels, Tensor)
            or next_sentence_labels.ndim != 1
        ):
            raise ValueError(
                "BERT-pretraining next-sentence labels must be a rank-1 tensor."
            )
        if next_sentence_labels.size(0) != input_ids.size(0):
            raise ValueError(
                "BERT-pretraining next-sentence labels must share the input batch "
                "dimension."
            )
        return batch

    def _validate_model_output(
        self,
        output: object,
        mlm_labels: Tensor,
        next_sentence_labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError(
                "BERT-pretraining outputs must be a three-item tuple containing "
                "(mlm_logits, nsp_logits, auxiliary_loss)."
            )
        mlm_logits, nsp_logits, auxiliary_loss = output
        if not isinstance(mlm_logits, Tensor) or mlm_logits.ndim != 3:
            raise ValueError(
                "BERT-pretraining MLM logits must be a rank-3 tensor with shape "
                "[batch, sequence, vocabulary]."
            )
        if mlm_logits.shape[:2] != mlm_labels.shape:
            raise ValueError(
                "BERT-pretraining MLM logits and labels must share batch and "
                "sequence dimensions."
            )
        if mlm_logits.size(-1) != self.vocab_size:
            raise ValueError(
                "BERT-pretraining MLM logits vocabulary dimension must equal "
                f"config.output_dim ({self.vocab_size}), received "
                f"{mlm_logits.size(-1)}."
            )
        if not isinstance(nsp_logits, Tensor) or nsp_logits.ndim != 2:
            raise ValueError(
                "BERT-pretraining NSP logits must be a rank-2 tensor with shape "
                "[batch, 2]."
            )
        if nsp_logits.size(0) != next_sentence_labels.size(0):
            raise ValueError(
                "BERT-pretraining NSP logits and labels must share the batch "
                "dimension."
            )
        if nsp_logits.size(1) != 2:
            raise ValueError(
                "BERT-pretraining NSP logits must contain exactly 2 classes."
            )
        resolved_auxiliary_loss = None
        if auxiliary_loss is not None:
            resolved_auxiliary_loss = self._auxiliary_loss.resolve(
                auxiliary_loss,
                reference=mlm_logits,
            )
        return mlm_logits, nsp_logits, auxiliary_loss, resolved_auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
