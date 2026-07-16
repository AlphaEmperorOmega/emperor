from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from ._metrics import SequenceClassifierMetricsLogger

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class SequenceClassifierExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.num_classes = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = SequenceClassifierMetricsLogger(self.num_classes)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, loss, logits, Y)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, loss, logits, Y)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, loss, logits, Y)
        return loss

    def _model_step(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        tokens, Y = batch
        tokens = tokens.to(self.device)
        output = self(tokens)
        if isinstance(output, tuple):
            logits, auxiliary_loss = output[0], output[-1]
            loss = self.loss_fn(logits, Y)
            if auxiliary_loss is not None and auxiliary_loss.item() != 0.0:
                loss = loss + auxiliary_loss
        else:
            logits = output
            loss = self.loss_fn(logits, Y)
        return loss, logits, Y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
