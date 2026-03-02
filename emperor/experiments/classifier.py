import torch
import torch.nn as nn
import torchmetrics

from torch import Tensor
from lightning import LightningModule
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ClassifierExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.num_classes = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = ClassifierMetricsLogger(self.num_classes)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, loss, logits, Y)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, loss, logits, Y)
        return loss

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, loss, logits, Y)
        return loss

    def _model_step(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        X, Y = batch
        output = self(X)
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


class ClassifierMetricsLogger(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        task = "multiclass"
        self.train_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.train_f1_score = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro")

        self.validation_accuracy = torchmetrics.Accuracy(
            task=task, num_classes=num_classes
        )
        self.validation_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes
        )

        self.test_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_f1_score = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro")

    def log_training_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.train_accuracy(logits, Y)
        f1score = self.train_f1_score(logits, Y)
        log_fn(
            {"train_loss": loss, "train_accuracy": accuracy, "train_f1_score": f1score},
            prog_bar=True,
        )

    def log_validation_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.validation_accuracy(logits, Y)
        f1score = self.validation_f1_score(logits, Y)
        log_fn(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_test_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.test_accuracy(logits, Y)
        f1score = self.test_f1_score(logits, Y)
        log_fn(
            {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1score},
        )
