from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from emperor.experiments._auxiliary_loss import AuxiliaryLoss
from emperor.experiments._config_validation import _ExperimentConfigValidator

from ._metrics import SequenceClassifierMetricsLogger
from ._records import SequenceClassifierBatch, SequenceClassifierStepOutput

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SequenceClassifierExperiment(LightningModule):
    VALIDATOR = _ExperimentConfigValidator

    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        resolved_config = self.VALIDATOR.resolve(
            cfg,
            "Sequence-classifier",
            minimum_output_dim=2,
        )
        self.learning_rate = resolved_config.learning_rate
        self.num_classes = resolved_config.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = SequenceClassifierMetricsLogger(self.num_classes)
        self._auxiliary_loss = AuxiliaryLoss("Sequence-classifier")

    def training_step(self, batch: SequenceClassifierBatch, batch_idx: int) -> Tensor:
        output = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, output)
        return output.total_loss

    def validation_step(
        self, batch: SequenceClassifierBatch, batch_idx: int
    ) -> Tensor:
        output = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, output)
        return output.total_loss

    def test_step(self, batch: SequenceClassifierBatch, batch_idx: int) -> Tensor:
        output = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, output)
        return output.total_loss

    def _model_step(
        self, batch: SequenceClassifierBatch
    ) -> SequenceClassifierStepOutput:
        tokens, labels = self._unpack_batch(batch)
        tokens = tokens.to(self.device)
        logits, resolved_auxiliary_loss = self._validate_model_output(
            self(tokens),
            labels,
        )
        task_loss = self.loss_fn(logits, labels)
        loss = task_loss
        if resolved_auxiliary_loss is not None:
            loss = task_loss + resolved_auxiliary_loss
        return SequenceClassifierStepOutput(loss, logits, labels)

    @staticmethod
    def _unpack_batch(batch: SequenceClassifierBatch) -> SequenceClassifierBatch:
        if len(batch) != 2:
            raise ValueError(
                "SequenceClassifierExperiment batches must contain (tokens, labels)."
            )
        tokens, labels = batch
        if not isinstance(tokens, Tensor) or tokens.ndim != 2:
            raise ValueError(
                "Sequence-classifier tokens must be a rank-2 tensor."
            )
        if not isinstance(labels, Tensor) or labels.ndim != 1:
            raise ValueError(
                "Sequence-classifier labels must be a rank-1 tensor."
            )
        if tokens.size(0) != labels.size(0):
            raise ValueError(
                "Sequence-classifier tokens and labels must share the batch "
                "dimension."
            )
        return tokens, labels

    def _validate_model_output(
        self,
        output: object,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Sequence-classifier tuple outputs must be a two-item tuple "
                    "containing (logits, auxiliary_loss)."
                )
            logits, auxiliary_loss = output
        else:
            logits = output
            auxiliary_loss = None
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError(
                "Sequence-classifier logits must be a rank-2 tensor with shape "
                "[batch, classes]."
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                "Sequence-classifier logits and labels must share the batch "
                "dimension."
            )
        if logits.size(1) != self.num_classes:
            raise ValueError(
                "Sequence-classifier logits class dimension must equal "
                f"config.output_dim ({self.num_classes}), received "
                f"{logits.size(1)}."
            )
        resolved_auxiliary_loss = None
        if auxiliary_loss is not None:
            resolved_auxiliary_loss = self._auxiliary_loss.resolve(
                auxiliary_loss,
                reference=logits,
            )
        return logits, resolved_auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
