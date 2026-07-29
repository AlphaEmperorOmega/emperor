from collections.abc import Callable

import torch.nn as nn
import torchmetrics

from ._records import SequenceClassifierStepOutput


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
        self, log_fn: Callable, output: SequenceClassifierStepOutput
    ) -> None:
        accuracy = self.train_accuracy(output.logits, output.labels)
        f1score = self.train_f1_score(output.logits, output.labels)
        log_fn(
            {
                "train/loss": output.total_loss,
                "train/accuracy": accuracy,
                "train/f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_validation_step(
        self, log_fn: Callable, output: SequenceClassifierStepOutput
    ) -> None:
        accuracy = self.validation_accuracy(output.logits, output.labels)
        f1score = self.validation_f1_score(output.logits, output.labels)
        log_fn(
            {
                "validation/loss": output.total_loss,
                "validation/accuracy": accuracy,
                "validation/f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_test_step(
        self, log_fn: Callable, output: SequenceClassifierStepOutput
    ) -> None:
        accuracy = self.test_accuracy(output.logits, output.labels)
        f1score = self.test_f1_score(output.logits, output.labels)
        log_fn(
            {
                "test/loss": output.total_loss,
                "test/accuracy": accuracy,
                "test/f1_score": f1score,
            },
        )
