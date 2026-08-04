from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class _ClassifierDiagnosticState:
    confidence_total: Tensor
    confidence_count: Tensor
    calibration_bin_confidence: Tensor
    calibration_bin_correct: Tensor
    calibration_bin_count: Tensor
    confusion_matrix: Tensor


class _ClassifierDiagnostics:
    def __init__(
        self,
        num_classes: int,
        confidence_bin_count: int,
        full_confusion_matrix_class_limit: int,
        top_confused_pair_limit: int,
    ) -> None:
        self._num_classes = num_classes
        self._confidence_bin_count = confidence_bin_count
        self._full_confusion_matrix_class_limit = full_confusion_matrix_class_limit
        self._top_confused_pair_limit = top_confused_pair_limit

    def update(
        self,
        state: _ClassifierDiagnosticState,
        logits: Tensor,
        labels: Tensor,
        targets: Tensor,
        predictions: Tensor,
    ) -> None:
        state.confusion_matrix.add_(
            self._batch_confusion_matrix(
                targets,
                predictions,
                dtype=state.confusion_matrix.dtype,
            )
        )
        self._update_confidence_totals(state, logits, labels)

    def reset(self, state: _ClassifierDiagnosticState) -> None:
        for value in (
            state.confusion_matrix,
            state.confidence_total,
            state.confidence_count,
            state.calibration_bin_confidence,
            state.calibration_bin_correct,
            state.calibration_bin_count,
        ):
            value.zero_()

    def per_class_metrics(
        self,
        prefix: str,
        state: _ClassifierDiagnosticState,
    ) -> dict[str, Tensor]:
        confusion_matrix = state.confusion_matrix
        if confusion_matrix.sum().item() == 0:
            return {}

        true_positive = confusion_matrix.diag()
        support = confusion_matrix.sum(dim=1)
        predicted = confusion_matrix.sum(dim=0)
        precision = self._safe_divide(true_positive, predicted)
        recall = self._safe_divide(true_positive, support)
        f1_score = self._safe_divide(2 * precision * recall, precision + recall)

        metrics: dict[str, Tensor] = {}
        for class_index in range(self._num_classes):
            class_prefix = f"{prefix}/per_class/class_{class_index}"
            metrics[f"{class_prefix}/accuracy"] = recall[class_index]
            metrics[f"{class_prefix}/precision"] = precision[class_index]
            metrics[f"{class_prefix}/recall"] = recall[class_index]
            metrics[f"{class_prefix}/f1_score"] = f1_score[class_index]
        return metrics

    def confusion_metrics(
        self,
        prefix: str,
        state: _ClassifierDiagnosticState,
    ) -> dict[str, Tensor]:
        confusion_matrix = state.confusion_matrix
        if confusion_matrix.sum().item() == 0:
            return {}

        if self._num_classes > self._full_confusion_matrix_class_limit:
            return self._top_confused_pair_metrics(prefix, confusion_matrix)

        support = confusion_matrix.sum(dim=1, keepdim=True)
        rate_matrix = self._safe_divide(confusion_matrix, support)
        metrics: dict[str, Tensor] = {}
        for true_class_index in range(self._num_classes):
            for predicted_class_index in range(self._num_classes):
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

    def confidence_metrics(
        self,
        prefix: str,
        state: _ClassifierDiagnosticState,
    ) -> dict[str, Tensor]:
        if state.confidence_count.item() == 0:
            return {}

        mean_confidence = state.confidence_total / state.confidence_count
        bin_confidence = self._safe_divide(
            state.calibration_bin_confidence,
            state.calibration_bin_count,
        )
        bin_accuracy = self._safe_divide(
            state.calibration_bin_correct,
            state.calibration_bin_count,
        )
        calibration_error = (
            state.calibration_bin_count
            / state.confidence_count
            * (bin_accuracy - bin_confidence).abs()
        ).sum()
        return {
            f"{prefix}/confidence/mean": mean_confidence,
            f"{prefix}/calibration/ece": calibration_error,
        }

    def _update_confidence_totals(
        self,
        state: _ClassifierDiagnosticState,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        confidence, correct, bin_indices = self._confidence_batch(
            logits,
            labels,
            device=state.confidence_total.device,
            dtype=state.confidence_total.dtype,
        )
        state.confidence_total.add_(confidence.sum().to(state.confidence_total.dtype))
        state.confidence_count.add_(
            torch.as_tensor(
                confidence.numel(),
                dtype=state.confidence_count.dtype,
                device=state.confidence_count.device,
            )
        )
        self._update_calibration_bins(state, confidence, correct, bin_indices)

    def _confidence_batch(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        device,
        dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        targets = labels.detach().to(device=device, dtype=torch.long).view(-1)
        probabilities = logits.detach().softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        confidence = confidence.to(device).view(-1)
        predictions = predictions.to(device).view(-1)
        correct = (predictions == targets).to(dtype)
        bin_indices = torch.clamp(
            (confidence * self._confidence_bin_count).long(),
            max=self._confidence_bin_count - 1,
        )
        return confidence, correct, bin_indices

    def _update_calibration_bins(
        self,
        state: _ClassifierDiagnosticState,
        confidence: Tensor,
        correct: Tensor,
        bin_indices: Tensor,
    ) -> None:
        state.calibration_bin_confidence.add_(
            torch.bincount(
                bin_indices,
                weights=confidence,
                minlength=self._confidence_bin_count,
            ).to(state.calibration_bin_confidence.dtype)
        )
        state.calibration_bin_correct.add_(
            torch.bincount(
                bin_indices,
                weights=correct,
                minlength=self._confidence_bin_count,
            ).to(state.calibration_bin_correct.dtype)
        )
        state.calibration_bin_count.add_(
            torch.bincount(
                bin_indices,
                minlength=self._confidence_bin_count,
            ).to(state.calibration_bin_count.dtype)
        )

    def _batch_confusion_matrix(
        self,
        targets: Tensor,
        predictions: Tensor,
        *,
        dtype,
    ) -> Tensor:
        indices = targets * self._num_classes + predictions
        return (
            torch.bincount(
                indices,
                minlength=self._num_classes * self._num_classes,
            )
            .reshape(self._num_classes, self._num_classes)
            .to(dtype)
        )

    def _top_confused_pair_metrics(
        self,
        prefix: str,
        confusion_matrix: Tensor,
    ) -> dict[str, Tensor]:
        if self._top_confused_pair_limit == 0:
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
        pair_limit = min(self._top_confused_pair_limit, nonzero_count)
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
                self._num_classes,
                rounding_mode="floor",
            )
            predicted_class_index = flat_index.remainder(self._num_classes)
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

    @staticmethod
    def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            torch.zeros_like(numerator),
        )
