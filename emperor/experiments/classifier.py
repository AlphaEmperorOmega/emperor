import torch
import torch.nn as nn
import torchmetrics

from torch import Tensor
from lightning import LightningModule
from typing import TYPE_CHECKING
from collections.abc import Callable
from emperor.experiments.monitor_policy import MonitorEmissionPolicy

if TYPE_CHECKING:
    from emperor.config import ModelConfig


DEFAULT_FULL_CONFUSION_MATRIX_CLASS_LIMIT = 20
DEFAULT_TOP_CONFUSED_PAIR_LIMIT = 50


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

    def on_train_epoch_start(self) -> None:
        self.metrics.reset_train_epoch()

    def on_train_epoch_end(self) -> None:
        self.metrics.log_train_epoch(self.log_dict)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, _ = batch
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, loss, logits, Y, X)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset_validation_epoch()

    def on_validation_epoch_end(self) -> None:
        self.metrics.log_validation_epoch_and_gap(self.log_dict)
        if not self.trainer.sanity_checking:
            self.metrics.log_best_validation(self.log_dict, self.current_epoch)
            self.metrics.log_validation_examples(self.logger, self.current_epoch)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, loss, logits, Y)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        health_metrics = self.optimizer_health_metrics(optimizer)
        if health_metrics:
            self.log_dict(
                health_metrics,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

    def _model_step(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        X, Y = batch
        output = self(X)
        if isinstance(output, tuple):
            logits, auxiliary_loss = output[0], output[-1]
            loss = self.loss_fn(logits, Y)
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss
        else:
            logits = output
            loss = self.loss_fn(logits, Y)
        return loss, logits, Y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def optimizer_health_metrics(self, optimizer) -> dict[str, Tensor]:
        totals = self._optimizer_health_totals(optimizer)
        parameter_norm = totals["parameter_square_total"].sqrt()
        gradient_norm = totals["gradient_square_total"].sqrt()
        update_norm = totals["update_square_total"].sqrt()
        update_to_weight_ratio = torch.where(
            parameter_norm > 0,
            update_norm / parameter_norm.clamp_min(1e-12),
            torch.zeros_like(parameter_norm),
        )
        return {
            "gradients/global_norm": gradient_norm,
            "parameters/global_norm": parameter_norm,
            "updates/update_to_weight_ratio": update_to_weight_ratio,
            "gradients/nan_count": totals["gradient_nan_count"],
            "gradients/inf_count": totals["gradient_inf_count"],
        }

    def _optimizer_health_totals(self, optimizer) -> dict[str, Tensor]:
        totals = {
            "parameter_square_total": torch.tensor(0.0, device=self.device),
            "gradient_square_total": torch.tensor(0.0, device=self.device),
            "update_square_total": torch.tensor(0.0, device=self.device),
            "gradient_nan_count": torch.tensor(0.0, device=self.device),
            "gradient_inf_count": torch.tensor(0.0, device=self.device),
        }
        for group in optimizer.param_groups:
            learning_rate = float(group.get("lr", self.learning_rate))
            for parameter in group.get("params", []):
                self._accumulate_optimizer_parameter_health(
                    totals,
                    learning_rate,
                    parameter,
                )
        return totals

    def _accumulate_optimizer_parameter_health(
        self,
        totals: dict[str, Tensor],
        learning_rate: float,
        parameter,
    ) -> None:
        if parameter is None:
            return
        parameter_data = parameter.detach()
        totals["parameter_square_total"] = totals[
            "parameter_square_total"
        ] + torch.nan_to_num(
            parameter_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).pow(2).sum()

        gradient = parameter.grad
        if gradient is None:
            return
        gradient_data = gradient.detach()
        finite_gradient = torch.nan_to_num(
            gradient_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        totals["gradient_square_total"] = (
            totals["gradient_square_total"] + finite_gradient.pow(2).sum()
        )
        totals["update_square_total"] = totals["update_square_total"] + (
            finite_gradient * learning_rate
        ).pow(2).sum()
        totals["gradient_nan_count"] = totals["gradient_nan_count"] + torch.isnan(
            gradient_data
        ).sum()
        totals["gradient_inf_count"] = totals["gradient_inf_count"] + torch.isinf(
            gradient_data
        ).sum()


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
        self._validation_examples = []
        self._emission_policy = MonitorEmissionPolicy()
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
        self, log_fn: Callable, loss: Tensor, logits: Tensor, Y: Tensor
    ) -> None:
        accuracy = self.train_accuracy(logits, Y)
        f1score = self.train_f1_score(logits, Y)
        self.update_train_epoch(loss, logits, Y)
        log_fn(
            {"train/loss": loss, "train/accuracy": accuracy, "train/f1_score": f1score},
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        logits: Tensor,
        Y: Tensor,
        examples: Tensor | None = None,
    ) -> None:
        accuracy = self.validation_accuracy(logits, Y)
        f1score = self.validation_f1_score(logits, Y)
        self.update_validation_epoch(loss, logits, Y)
        self.update_validation_examples(examples, logits, Y)
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

    def update_train_epoch(self, loss: Tensor, logits: Tensor, Y: Tensor) -> None:
        self._update_stage_epoch("train", loss, logits, Y)

    def update_validation_epoch(self, loss: Tensor, logits: Tensor, Y: Tensor) -> None:
        self._update_stage_epoch("validation", loss, logits, Y)

    def _update_stage_epoch(
        self,
        prefix: str,
        loss: Tensor,
        logits: Tensor,
        Y: Tensor,
    ) -> None:
        self._update_epoch_totals(
            loss,
            logits,
            Y,
            getattr(self, f"_{prefix}_loss_total"),
            getattr(self, f"_{prefix}_correct"),
            getattr(self, f"_{prefix}_count"),
            getattr(self, f"_{prefix}_confusion_matrix"),
        )
        self._update_confidence_totals(
            logits,
            Y,
            getattr(self, f"_{prefix}_confidence_total"),
            getattr(self, f"_{prefix}_confidence_count"),
            getattr(self, f"_{prefix}_calibration_bin_confidence"),
            getattr(self, f"_{prefix}_calibration_bin_correct"),
            getattr(self, f"_{prefix}_calibration_bin_count"),
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
        return self._epoch_metrics(
            "train",
            self._train_loss_total,
            self._train_correct,
            self._train_count,
        )

    def train_per_class_epoch_metrics(self) -> dict[str, Tensor]:
        return self._per_class_epoch_metrics("train", self._train_confusion_matrix)

    def train_confusion_matrix_epoch_metrics(self) -> dict[str, Tensor]:
        return self._confusion_matrix_epoch_metrics(
            "train",
            self._train_confusion_matrix,
        )

    def train_confidence_epoch_metrics(self) -> dict[str, Tensor]:
        return self._confidence_epoch_metrics(
            "train",
            self._train_confidence_total,
            self._train_confidence_count,
            self._train_calibration_bin_confidence,
            self._train_calibration_bin_correct,
            self._train_calibration_bin_count,
        )

    def validation_epoch_metrics(self) -> dict[str, Tensor]:
        return self._epoch_metrics(
            "validation",
            self._validation_loss_total,
            self._validation_correct,
            self._validation_count,
        )

    def validation_per_class_epoch_metrics(self) -> dict[str, Tensor]:
        return self._per_class_epoch_metrics(
            "validation",
            self._validation_confusion_matrix,
        )

    def validation_confusion_matrix_epoch_metrics(self) -> dict[str, Tensor]:
        return self._confusion_matrix_epoch_metrics(
            "validation",
            self._validation_confusion_matrix,
        )

    def validation_confidence_epoch_metrics(self) -> dict[str, Tensor]:
        return self._confidence_epoch_metrics(
            "validation",
            self._validation_confidence_total,
            self._validation_confidence_count,
            self._validation_calibration_bin_confidence,
            self._validation_calibration_bin_correct,
            self._validation_calibration_bin_count,
        )

    def train_validation_gap_metrics(self) -> dict[str, Tensor]:
        train_metrics = self.train_epoch_metrics()
        validation_metrics = self.validation_epoch_metrics()
        if not train_metrics or not validation_metrics:
            return {}
        return {
            "gap/accuracy": (
                train_metrics["train/accuracy_epoch"]
                - validation_metrics["validation/accuracy_epoch"]
            ),
            "gap/loss": (
                validation_metrics["validation/loss_epoch"]
                - train_metrics["train/loss_epoch"]
            ),
        }

    def reset_train_epoch(self) -> None:
        self.train_accuracy.reset()
        self.train_f1_score.reset()
        self._reset_epoch_buffers("train")

    def reset_validation_epoch(self) -> None:
        self.validation_accuracy.reset()
        self.validation_f1_score.reset()
        self._reset_epoch_buffers("validation")
        self._validation_examples = []

    def _reset_epoch_buffers(self, prefix: str) -> None:
        for suffix in (
            "loss_total",
            "correct",
            "count",
            "confusion_matrix",
            "confidence_total",
            "confidence_count",
            "calibration_bin_confidence",
            "calibration_bin_correct",
            "calibration_bin_count",
        ):
            getattr(self, f"_{prefix}_{suffix}").zero_()

    def _update_epoch_totals(
        self,
        loss: Tensor,
        logits: Tensor,
        Y: Tensor,
        loss_total: Tensor,
        correct_total: Tensor,
        count_total: Tensor,
        confusion_matrix: Tensor,
    ) -> None:
        targets = Y.detach().to(device=count_total.device, dtype=torch.long).view(-1)
        predictions = logits.detach().argmax(dim=1).to(count_total.device).view(-1)
        batch_count = torch.as_tensor(
            targets.numel(),
            dtype=loss_total.dtype,
            device=loss_total.device,
        )
        correct = (predictions == targets).sum().to(
            dtype=correct_total.dtype,
            device=correct_total.device,
        )
        loss_total.add_(loss.detach().to(loss_total.device) * batch_count)
        correct_total.add_(correct)
        count_total.add_(batch_count)
        confusion_matrix.add_(self._batch_confusion_matrix(targets, predictions))

    def _update_confidence_totals(
        self,
        logits: Tensor,
        Y: Tensor,
        confidence_total: Tensor,
        confidence_count: Tensor,
        bin_confidence_total: Tensor,
        bin_correct_total: Tensor,
        bin_count_total: Tensor,
    ) -> None:
        confidence, correct, bin_indices = self._confidence_batch(
            logits,
            Y,
            device=confidence_total.device,
            dtype=confidence_total.dtype,
        )
        confidence_total.add_(confidence.sum().to(confidence_total.dtype))
        confidence_count.add_(
            torch.as_tensor(
                confidence.numel(),
                dtype=confidence_count.dtype,
                device=confidence_count.device,
            )
        )
        self._update_calibration_bins(
            confidence,
            correct,
            bin_indices,
            bin_confidence_total,
            bin_correct_total,
            bin_count_total,
        )

    def _confidence_batch(
        self,
        logits: Tensor,
        Y: Tensor,
        *,
        device,
        dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        targets = Y.detach().to(device=device, dtype=torch.long).view(-1)
        probabilities = logits.detach().softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        confidence = confidence.to(device).view(-1)
        predictions = predictions.to(device).view(-1)
        correct = (predictions == targets).to(dtype)
        bin_indices = torch.clamp(
            (confidence * self.confidence_bin_count).long(),
            max=self.confidence_bin_count - 1,
        )
        return confidence, correct, bin_indices

    def _update_calibration_bins(
        self,
        confidence: Tensor,
        correct: Tensor,
        bin_indices: Tensor,
        bin_confidence_total: Tensor,
        bin_correct_total: Tensor,
        bin_count_total: Tensor,
    ) -> None:
        bin_confidence_total.add_(
            torch.bincount(
                bin_indices,
                weights=confidence,
                minlength=self.confidence_bin_count,
            ).to(bin_confidence_total.dtype)
        )
        bin_correct_total.add_(
            torch.bincount(
                bin_indices,
                weights=correct,
                minlength=self.confidence_bin_count,
            ).to(bin_correct_total.dtype)
        )
        bin_count_total.add_(
            torch.bincount(
                bin_indices,
                minlength=self.confidence_bin_count,
            ).to(bin_count_total.dtype)
        )

    def update_validation_examples(
        self,
        examples: Tensor | None,
        logits: Tensor,
        Y: Tensor,
    ) -> None:
        if examples is None or examples.ndim != 4:
            return
        if examples.size(1) not in (1, 3):
            return

        probabilities = logits.detach().softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        targets = Y.detach().to(predictions.device)
        wrong_indices = (predictions != targets).nonzero(as_tuple=False).flatten()

        for index in wrong_indices.tolist():
            image = examples[index].detach().cpu()
            self._validation_examples.append(
                {
                    "confidence": float(confidence[index].detach().cpu().item()),
                    "image": image,
                    "prediction": int(predictions[index].detach().cpu().item()),
                    "target": int(targets[index].detach().cpu().item()),
                }
            )
        self._validation_examples.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )
        del self._validation_examples[self.validation_example_limit :]

    def _epoch_metrics(
        self,
        prefix: str,
        loss_total: Tensor,
        correct_total: Tensor,
        count_total: Tensor,
    ) -> dict[str, Tensor]:
        if count_total.item() == 0:
            return {}
        return {
            f"{prefix}/loss_epoch": loss_total / count_total,
            f"{prefix}/accuracy_epoch": correct_total / count_total,
        }

    def log_best_validation(self, log_fn: Callable, epoch: int) -> None:
        best_metrics = self.update_best_validation_metrics(epoch)
        if best_metrics:
            log_fn(best_metrics, prog_bar=True, on_step=False, on_epoch=True)

    def update_best_validation_metrics(self, epoch: int) -> dict[str, Tensor]:
        validation_metrics = self.validation_epoch_metrics()
        if not validation_metrics:
            return {}

        validation_accuracy = validation_metrics["validation/accuracy_epoch"].detach()
        validation_loss = validation_metrics["validation/loss_epoch"].detach()
        epoch_tensor = torch.as_tensor(
            float(epoch),
            dtype=self._best_validation_accuracy.dtype,
            device=self._best_validation_accuracy.device,
        )
        if validation_accuracy > self._best_validation_accuracy:
            self._best_validation_accuracy.copy_(validation_accuracy)
            self._best_validation_accuracy_epoch.copy_(epoch_tensor)
        if validation_loss < self._best_validation_loss:
            self._best_validation_loss.copy_(validation_loss)
            self._best_validation_loss_epoch.copy_(epoch_tensor)

        return {
            "best_validation/accuracy": self._best_validation_accuracy,
            "best_validation/loss": self._best_validation_loss,
            "best_validation/epoch": self._best_validation_accuracy_epoch,
            "best_validation/accuracy_epoch": self._best_validation_accuracy_epoch,
            "best_validation/loss_epoch": self._best_validation_loss_epoch,
        }

    def _batch_confusion_matrix(self, targets: Tensor, predictions: Tensor) -> Tensor:
        indices = targets * self.num_classes + predictions
        return torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes).to(self._train_loss_total.dtype)

    def _per_class_epoch_metrics(
        self,
        prefix: str,
        confusion_matrix: Tensor,
    ) -> dict[str, Tensor]:
        if confusion_matrix.sum().item() == 0:
            return {}

        true_positive = confusion_matrix.diag()
        support = confusion_matrix.sum(dim=1)
        predicted = confusion_matrix.sum(dim=0)
        precision = self._safe_divide(true_positive, predicted)
        recall = self._safe_divide(true_positive, support)
        f1_score = self._safe_divide(2 * precision * recall, precision + recall)

        metrics: dict[str, Tensor] = {}
        for class_index in range(self.num_classes):
            class_prefix = f"{prefix}/per_class/class_{class_index}"
            metrics[f"{class_prefix}/accuracy"] = recall[class_index]
            metrics[f"{class_prefix}/precision"] = precision[class_index]
            metrics[f"{class_prefix}/recall"] = recall[class_index]
            metrics[f"{class_prefix}/f1_score"] = f1_score[class_index]
        return metrics

    def _confusion_matrix_epoch_metrics(
        self,
        prefix: str,
        confusion_matrix: Tensor,
    ) -> dict[str, Tensor]:
        if confusion_matrix.sum().item() == 0:
            return {}

        if self.num_classes > self.full_confusion_matrix_class_limit:
            return self._top_confused_pair_metrics(prefix, confusion_matrix)

        support = confusion_matrix.sum(dim=1, keepdim=True)
        rate_matrix = self._safe_divide(confusion_matrix, support)
        metrics: dict[str, Tensor] = {}
        for true_class_index in range(self.num_classes):
            for predicted_class_index in range(self.num_classes):
                cell_prefix = (
                    f"{prefix}/confusion_matrix"
                    f"/true_class_{true_class_index}"
                    f"/predicted_class_{predicted_class_index}"
                )
                metrics[f"{cell_prefix}/count"] = confusion_matrix[
                    true_class_index,
                    predicted_class_index,
                ]
                metrics[f"{cell_prefix}/rate"] = rate_matrix[
                    true_class_index,
                    predicted_class_index,
                ]
        return metrics

    def _top_confused_pair_metrics(
        self,
        prefix: str,
        confusion_matrix: Tensor,
    ) -> dict[str, Tensor]:
        if self.top_confused_pair_limit == 0:
            return {}
        top_pairs = self._top_confused_pairs(confusion_matrix)
        if top_pairs is None:
            return {}

        values, flat_indices = top_pairs
        support = confusion_matrix.sum(dim=1, keepdim=True)
        rate_matrix = self._safe_divide(confusion_matrix, support)
        return self._top_confused_pair_payload(
            prefix,
            confusion_matrix,
            rate_matrix,
            values,
            flat_indices,
        )

    def _top_confused_pairs(
        self,
        confusion_matrix: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        off_diagonal = confusion_matrix.clone()
        off_diagonal.fill_diagonal_(0)
        nonzero_count = int((off_diagonal > 0).sum().item())
        if nonzero_count == 0:
            return None
        pair_limit = min(self.top_confused_pair_limit, nonzero_count)
        return torch.topk(off_diagonal.flatten(), pair_limit)

    def _top_confused_pair_payload(
        self,
        prefix: str,
        confusion_matrix: Tensor,
        rate_matrix: Tensor,
        values: Tensor,
        flat_indices: Tensor,
    ) -> dict[str, Tensor]:
        metrics: dict[str, Tensor] = {}
        for rank, (value, flat_index) in enumerate(
            zip(values, flat_indices, strict=True),
            start=1,
        ):
            if value.item() <= 0:
                continue
            true_class_index = torch.div(
                flat_index,
                self.num_classes,
                rounding_mode="floor",
            )
            predicted_class_index = flat_index.remainder(self.num_classes)
            pair_prefix = f"{prefix}/confusion_top_pairs/rank_{rank}"
            metrics[f"{pair_prefix}/count"] = value
            metrics[f"{pair_prefix}/rate"] = rate_matrix[
                true_class_index,
                predicted_class_index,
            ]
            metrics[f"{pair_prefix}/true_class"] = true_class_index.to(
                dtype=confusion_matrix.dtype
            )
            metrics[f"{pair_prefix}/predicted_class"] = predicted_class_index.to(
                dtype=confusion_matrix.dtype
            )
        return metrics

    def _confidence_epoch_metrics(
        self,
        prefix: str,
        confidence_total: Tensor,
        confidence_count: Tensor,
        bin_confidence_total: Tensor,
        bin_correct_total: Tensor,
        bin_count_total: Tensor,
    ) -> dict[str, Tensor]:
        if confidence_count.item() == 0:
            return {}

        mean_confidence = confidence_total / confidence_count
        bin_confidence = self._safe_divide(bin_confidence_total, bin_count_total)
        bin_accuracy = self._safe_divide(bin_correct_total, bin_count_total)
        calibration_error = (
            bin_count_total
            / confidence_count
            * (bin_accuracy - bin_confidence).abs()
        ).sum()
        return {
            f"{prefix}/confidence/mean": mean_confidence,
            f"{prefix}/calibration/ece": calibration_error,
        }

    def log_validation_examples(self, logger, epoch: int) -> None:
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not self._validation_examples:
            return

        image_grid = self._validation_example_grid()
        if image_grid is not None:
            self._emission_policy.emit_image(
                experiment,
                "validation/examples/most_confident_wrong",
                image_grid,
                global_step=epoch,
                module_key="validation",
            )

        add_text = getattr(experiment, "add_text", None)
        if callable(add_text):
            lines = [
                (
                    f"{index}. true={example['target']} "
                    f"predicted={example['prediction']} "
                    f"confidence={example['confidence']:.4f}"
                )
                for index, example in enumerate(self._validation_examples, start=1)
            ]
            add_text(
                "validation/examples/most_confident_wrong_labels",
                "\n".join(lines),
                global_step=epoch,
            )

    def _validation_example_grid(self) -> Tensor | None:
        images = [example["image"] for example in self._validation_examples]
        if not images:
            return None

        image_batch = torch.stack(images).float()
        image_min = image_batch.amin(dim=(1, 2, 3), keepdim=True)
        image_max = image_batch.amax(dim=(1, 2, 3), keepdim=True)
        image_range = image_max - image_min
        image_batch = torch.where(
            image_range > 0,
            (image_batch - image_min) / image_range.clamp_min(1e-12),
            image_batch.clamp(0, 1),
        )
        if image_batch.size(1) == 1:
            image_batch = image_batch.repeat(1, 3, 1, 1)

        image_count, channels, height, width = image_batch.shape
        columns = min(4, image_count)
        rows = (image_count + columns - 1) // columns
        grid = image_batch.new_zeros(channels, rows * height, columns * width)
        for index, image in enumerate(image_batch):
            row = index // columns
            column = index % columns
            grid[
                :,
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ] = image
        return grid

    def _safe_divide(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            torch.zeros_like(numerator),
        )
