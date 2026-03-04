import math
import torch
import torch.nn as nn
import torchmetrics

from torch import Tensor
from lightning import LightningModule
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LanguageModelExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = LanguageModelMetricsLogger()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, loss)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, loss)
        return loss

    def _model_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        tokens, targets = batch
        tokens = tokens.to(self.device)
        targets = targets.to(self.device)
        output = self(tokens)
        # output: (batch, seq_len, vocab_size) → CrossEntropyLoss expects (batch, vocab_size, seq_len)
        loss = self.loss_fn(output.transpose(1, 2), targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LanguageModelMetricsLogger(nn.Module):
    def __init__(self):
        super().__init__()

    def log_training_step(self, log_fn: Callable, loss: Tensor) -> None:
        log_fn(
            {"train_loss": loss, "train_perplexity": math.exp(loss.item())},
            prog_bar=True,
        )

    def log_validation_step(self, log_fn: Callable, loss: Tensor) -> None:
        log_fn(
            {"validation_loss": loss, "validation_perplexity": math.exp(loss.item())},
            prog_bar=True,
        )

    def log_test_step(self, log_fn: Callable, loss: Tensor) -> None:
        log_fn(
            {"test_loss": loss, "test_perplexity": math.exp(loss.item())},
        )
