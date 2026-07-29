from collections.abc import Callable

import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor

from ._diagnostics import _ClassifierDiagnostics, _ClassifierDiagnosticState
from ._epoch_metrics import (
    _BestValidationState,
    _ClassifierEpochMetrics,
    _ClassifierEpochState,
)
from ._records import ClassifierStepOutput
from ._validation_examples import _ClassifierValidationExamples

DEFAULT_FULL_CONFUSION_MATRIX_CLASS_LIMIT = 20
DEFAULT_TOP_CONFUSED_PAIR_LIMIT = 50


class ClassifierMetricsLogger(nn.Module):
    def __init__(
        self,
        num_classes: int,
        confidence_bin_count: int = 10,
        validation_example_limit: int = 16,
        full_confusion_matrix_class_limit: int = (
            DEFAULT_FULL_CONFUSION_MATRIX_CLASS_LIMIT
        ),
        top_confused_pair_limit: int = DEFAULT_TOP_CONFUSED_PAIR_LIMIT,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.confidence_bin_count = confidence_bin_count
        self.validation_example_limit = validation_example_limit
        self.full_confusion_matrix_class_limit = max(
            0,
            int(full_confusion_matrix_class_limit),
        )
        self.top_confused_pair_limit = max(0, int(top_confused_pair_limit))
        self._epoch_metrics_owner = _ClassifierEpochMetrics()
        self._diagnostics_owner = _ClassifierDiagnostics(
            num_classes=self.num_classes,
            confidence_bin_count=self.confidence_bin_count,
            full_confusion_matrix_class_limit=(
                self.full_confusion_matrix_class_limit
            ),
            top_confused_pair_limit=self.top_confused_pair_limit,
        )
        self._validation_examples_owner = _ClassifierValidationExamples(
            self.validation_example_limit
        )
        task = "multiclass"
        self._register_epoch_buffers("train")
        self._register_epoch_buffers("validation")
        self._register_best_validation_buffers()
        self.train_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.train_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        )

        self.validation_accuracy = torchmetrics.Accuracy(
            task=task, num_classes=num_classes
        )
        self.validation_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes
        )

        self.test_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_f1_score = torchmetrics.F1Score(
            task=task, num_classes=num_classes, average="macro"
        )

    def _register_epoch_buffers(self, prefix: str) -> None:
        self.register_buffer(
            f"_{prefix}_loss_total", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(f"_{prefix}_correct", torch.tensor(0.0), persistent=False)
        self.register_buffer(f"_{prefix}_count", torch.tensor(0.0), persistent=False)
        self.register_buffer(
            f"_{prefix}_confidence_total", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            f"_{prefix}_confidence_count", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            f"_{prefix}_calibration_bin_confidence",
            torch.zeros(self.confidence_bin_count),
            persistent=False,
        )
        self.register_buffer(
            f"_{prefix}_calibration_bin_correct",
            torch.zeros(self.confidence_bin_count),
            persistent=False,
        )
        self.register_buffer(
            f"_{prefix}_calibration_bin_count",
            torch.zeros(self.confidence_bin_count),
            persistent=False,
        )
        self.register_buffer(
            f"_{prefix}_confusion_matrix",
            torch.zeros(self.num_classes, self.num_classes),
            persistent=False,
        )

    def _register_best_validation_buffers(self) -> None:
        self.register_buffer(
            "_best_validation_accuracy",
            torch.tensor(float("-inf")),
            persistent=False,
        )
        self.register_buffer(
            "_best_validation_loss",
            torch.tensor(float("inf")),
            persistent=False,
        )
        self.register_buffer(
            "_best_validation_accuracy_epoch", torch.tensor(-1.0), persistent=False
        )
        self.register_buffer(
            "_best_validation_loss_epoch", torch.tensor(-1.0), persistent=False
        )

    def log_training_step(
        self, log_fn: Callable, output: ClassifierStepOutput
    ) -> None:
        accuracy = self.train_accuracy(output.logits, output.labels)
        f1score = self.train_f1_score(output.logits, output.labels)
        self.update_train_epoch(output.total_loss, output.logits, output.labels)
        log_fn(
            {
                "train/loss": output.total_loss,
                "train/accuracy": accuracy,
                "train/f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        output: ClassifierStepOutput,
        examples: Tensor | None = None,
    ) -> None:
        accuracy = self.validation_accuracy(output.logits, output.labels)
        f1score = self.validation_f1_score(output.logits, output.labels)
        self.update_validation_epoch(
            output.total_loss,
            output.logits,
            output.labels,
        )
        self.update_validation_examples(examples, output.logits, output.labels)
        log_fn(
            {
                "validation/loss": output.total_loss,
                "validation/accuracy": accuracy,
                "validation/f1_score": f1score,
            },
            prog_bar=True,
        )

    def log_test_step(
        self, log_fn: Callable, output: ClassifierStepOutput
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

    def update_train_epoch(self, loss: Tensor, logits: Tensor, labels: Tensor) -> None:
        batch_predictions = self._epoch_metrics_owner.update(
            self._epoch_state("train"),
            loss,
            logits,
            labels,
        )
        self._diagnostics_owner.update(
            self._diagnostic_state("train"),
            logits,
            labels,
            batch_predictions.targets,
            batch_predictions.predictions,
        )

    def update_validation_epoch(
        self,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        batch_predictions = self._epoch_metrics_owner.update(
            self._epoch_state("validation"),
            loss,
            logits,
            labels,
        )
        self._diagnostics_owner.update(
            self._diagnostic_state("validation"),
            logits,
            labels,
            batch_predictions.targets,
            batch_predictions.predictions,
        )

    def log_train_epoch(self, log_fn: Callable) -> None:
        train_metrics = self.train_epoch_metrics()
        if train_metrics:
            log_fn(train_metrics, prog_bar=True, on_step=False, on_epoch=True)
        diagnostic_metrics = {
            **self.train_confidence_epoch_metrics(),
            **self.train_per_class_epoch_metrics(),
            **self.train_confusion_matrix_epoch_metrics(),
        }
        if diagnostic_metrics:
            log_fn(diagnostic_metrics, prog_bar=False, on_step=False, on_epoch=True)

    def log_validation_epoch_and_gap(self, log_fn: Callable) -> None:
        validation_metrics = self.validation_epoch_metrics()
        gap_metrics = self.train_validation_gap_metrics()
        payload = {**validation_metrics, **gap_metrics}
        if payload:
            log_fn(payload, prog_bar=True, on_step=False, on_epoch=True)
        diagnostic_metrics = {
            **self.validation_confidence_epoch_metrics(),
            **self.validation_per_class_epoch_metrics(),
            **self.validation_confusion_matrix_epoch_metrics(),
        }
        if diagnostic_metrics:
            log_fn(diagnostic_metrics, prog_bar=False, on_step=False, on_epoch=True)

    def train_epoch_metrics(self) -> dict[str, Tensor]:
        return self._epoch_metrics_owner.metrics("train", self._epoch_state("train"))

    def train_per_class_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.per_class_metrics(
            "train",
            self._diagnostic_state("train"),
        )

    def train_confusion_matrix_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.confusion_metrics(
            "train",
            self._diagnostic_state("train"),
        )

    def train_confidence_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.confidence_metrics(
            "train",
            self._diagnostic_state("train"),
        )

    def validation_epoch_metrics(self) -> dict[str, Tensor]:
        return self._epoch_metrics_owner.metrics(
            "validation",
            self._epoch_state("validation"),
        )

    def validation_per_class_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.per_class_metrics(
            "validation",
            self._diagnostic_state("validation"),
        )

    def validation_confusion_matrix_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.confusion_metrics(
            "validation",
            self._diagnostic_state("validation"),
        )

    def validation_confidence_epoch_metrics(self) -> dict[str, Tensor]:
        return self._diagnostics_owner.confidence_metrics(
            "validation",
            self._diagnostic_state("validation"),
        )

    def train_validation_gap_metrics(self) -> dict[str, Tensor]:
        return self._epoch_metrics_owner.gap_metrics(
            self._epoch_state("train"),
            self._epoch_state("validation"),
        )

    def reset_train_epoch(self) -> None:
        self.train_accuracy.reset()
        self.train_f1_score.reset()
        self._epoch_metrics_owner.reset(self._epoch_state("train"))
        self._diagnostics_owner.reset(self._diagnostic_state("train"))

    def reset_validation_epoch(self) -> None:
        self.validation_accuracy.reset()
        self.validation_f1_score.reset()
        self._epoch_metrics_owner.reset(self._epoch_state("validation"))
        self._diagnostics_owner.reset(self._diagnostic_state("validation"))
        self._validation_examples_owner.reset()

    def update_validation_examples(
        self,
        examples: Tensor | None,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        self._validation_examples_owner.update(examples, logits, labels)

    def log_best_validation(self, log_fn: Callable, epoch: int) -> None:
        best_metrics = self.update_best_validation_metrics(epoch)
        if best_metrics:
            log_fn(best_metrics, prog_bar=True, on_step=False, on_epoch=True)

    def update_best_validation_metrics(self, epoch: int) -> dict[str, Tensor]:
        return self._epoch_metrics_owner.update_best_validation(
            self._epoch_state("validation"),
            self._best_validation_state(),
            epoch,
        )

    def log_validation_examples(self, logger, epoch: int) -> None:
        self._validation_examples_owner.emit(logger, epoch)

    def _epoch_state(self, prefix: str) -> _ClassifierEpochState:
        return _ClassifierEpochState(
            loss_total=getattr(self, f"_{prefix}_loss_total"),
            correct=getattr(self, f"_{prefix}_correct"),
            count=getattr(self, f"_{prefix}_count"),
        )

    def _diagnostic_state(self, prefix: str) -> _ClassifierDiagnosticState:
        return _ClassifierDiagnosticState(
            confidence_total=getattr(self, f"_{prefix}_confidence_total"),
            confidence_count=getattr(self, f"_{prefix}_confidence_count"),
            calibration_bin_confidence=getattr(
                self,
                f"_{prefix}_calibration_bin_confidence",
            ),
            calibration_bin_correct=getattr(
                self,
                f"_{prefix}_calibration_bin_correct",
            ),
            calibration_bin_count=getattr(
                self,
                f"_{prefix}_calibration_bin_count",
            ),
            confusion_matrix=getattr(self, f"_{prefix}_confusion_matrix"),
        )

    def _best_validation_state(self) -> _BestValidationState:
        return _BestValidationState(
            accuracy=self._best_validation_accuracy,
            loss=self._best_validation_loss,
            accuracy_epoch=self._best_validation_accuracy_epoch,
            loss_epoch=self._best_validation_loss_epoch,
        )
