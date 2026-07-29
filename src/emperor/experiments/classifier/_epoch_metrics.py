from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class _ClassifierEpochState:
    loss_total: Tensor
    correct: Tensor
    count: Tensor


@dataclass(frozen=True)
class _ClassifierBatchPredictions:
    targets: Tensor
    predictions: Tensor


@dataclass(frozen=True)
class _BestValidationState:
    accuracy: Tensor
    loss: Tensor
    accuracy_epoch: Tensor
    loss_epoch: Tensor


class _ClassifierEpochMetrics:
    def update(
        self,
        state: _ClassifierEpochState,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
    ) -> _ClassifierBatchPredictions:
        targets = labels.detach().to(
            device=state.count.device,
            dtype=torch.long,
        ).view(-1)
        predictions = logits.detach().argmax(dim=1).to(state.count.device).view(-1)
        batch_count = torch.as_tensor(
            targets.numel(),
            dtype=state.loss_total.dtype,
            device=state.loss_total.device,
        )
        correct = (
            (predictions == targets)
            .sum()
            .to(
                dtype=state.correct.dtype,
                device=state.correct.device,
            )
        )
        state.loss_total.add_(loss.detach().to(state.loss_total.device) * batch_count)
        state.correct.add_(correct)
        state.count.add_(batch_count)
        return _ClassifierBatchPredictions(targets, predictions)

    def reset(self, state: _ClassifierEpochState) -> None:
        state.loss_total.zero_()
        state.correct.zero_()
        state.count.zero_()

    def metrics(
        self,
        prefix: str,
        state: _ClassifierEpochState,
    ) -> dict[str, Tensor]:
        if state.count.item() == 0:
            return {}
        return {
            f"{prefix}/loss_epoch": state.loss_total / state.count,
            f"{prefix}/accuracy_epoch": state.correct / state.count,
        }

    def gap_metrics(
        self,
        train_state: _ClassifierEpochState,
        validation_state: _ClassifierEpochState,
    ) -> dict[str, Tensor]:
        train_metrics = self.metrics("train", train_state)
        validation_metrics = self.metrics("validation", validation_state)
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

    def update_best_validation(
        self,
        validation_state: _ClassifierEpochState,
        best_state: _BestValidationState,
        epoch: int,
    ) -> dict[str, Tensor]:
        validation_metrics = self.metrics("validation", validation_state)
        if not validation_metrics:
            return {}

        validation_accuracy = validation_metrics["validation/accuracy_epoch"].detach()
        validation_loss = validation_metrics["validation/loss_epoch"].detach()
        epoch_tensor = torch.as_tensor(
            float(epoch),
            dtype=best_state.accuracy.dtype,
            device=best_state.accuracy.device,
        )
        if validation_accuracy > best_state.accuracy:
            best_state.accuracy.copy_(validation_accuracy)
            best_state.accuracy_epoch.copy_(epoch_tensor)
        if validation_loss < best_state.loss:
            best_state.loss.copy_(validation_loss)
            best_state.loss_epoch.copy_(epoch_tensor)

        return {
            "best_validation/accuracy": best_state.accuracy,
            "best_validation/loss": best_state.loss,
            "best_validation/epoch": best_state.accuracy_epoch,
            "best_validation/accuracy_epoch": best_state.accuracy_epoch,
            "best_validation/loss_epoch": best_state.loss_epoch,
        }
