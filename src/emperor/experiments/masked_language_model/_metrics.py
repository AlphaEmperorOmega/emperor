import math
from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


class MaskedLanguageModelMetricsLogger(nn.Module):
    def __init__(self):
        super().__init__()

    def log_training_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload("train", loss, logits, labels, auxiliary_loss),
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload("validation", loss, logits, labels, auxiliary_loss),
            prog_bar=True,
        )

    def log_test_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload("test", loss, logits, labels, auxiliary_loss),
        )

    def _payload(
        self,
        stage: str,
        loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None,
    ) -> dict[str, Tensor | float]:
        payload: dict[str, Tensor | float] = {
            f"{stage}/loss": loss,
            f"{stage}/perplexity": math.exp(loss.item()),
            f"{stage}/masked/accuracy": self._masked_accuracy(logits, labels),
            f"{stage}/masked/top_5_accuracy": self._masked_top_k_accuracy(
                logits,
                labels,
                k=5,
            ),
        }
        if auxiliary_loss is not None:
            payload[f"{stage}/auxiliary/loss"] = auxiliary_loss
        return payload

    def _masked_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        mask = labels != -100
        if not bool(mask.any().item()):
            return logits.new_zeros(())
        predictions = logits.argmax(dim=-1)
        return (predictions[mask] == labels[mask]).float().mean()

    def _masked_top_k_accuracy(
        self,
        logits: Tensor,
        labels: Tensor,
        k: int,
    ) -> Tensor:
        mask = labels != -100
        if not bool(mask.any().item()):
            return logits.new_zeros(())
        top_k = logits.topk(min(k, logits.size(-1)), dim=-1).indices
        matches = top_k[mask].eq(labels[mask].unsqueeze(-1)).any(dim=-1)
        return matches.float().mean()
