from collections.abc import Callable

import torch.nn as nn
import torchmetrics
from torch import Tensor


class SequenceClassifierMetricsLogger(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        task = "multiclass"
        self.train_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.train_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        )

        self.validation_accuracy = torchmetrics.Accuracy(
            task=task, num_classes=num_classes
        )
        self.validation_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        )

        self.test_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        )

    def log_training_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.train_accuracy(logits, Y)
        f1score = self.train_f1_score(logits, Y)
        log_fn(
            {"train/loss": loss, "train/accuracy": accuracy, "train/f1_score": f1score},
            prog_bar=True,
        )

    def log_validation_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.validation_accuracy(logits, Y)
        f1score = self.validation_f1_score(logits, Y)
        log_fn(
            {
                "validation/loss": loss,
                "validation/accuracy": accuracy,
                "validation/f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_test_step(
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.test_accuracy(logits, Y)
        f1score = self.test_f1_score(logits, Y)
        log_fn(
            {"test/loss": loss, "test/accuracy": accuracy, "test/f1_score": f1score},
        )
