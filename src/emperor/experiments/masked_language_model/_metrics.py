from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from emperor.experiments._perplexity import Perplexity


class MaskedLanguageModelMetricsLogger(nn.Module):
    def __init__(self):
        super().__init__()
        self._perplexity = Perplexity()

    def log_training_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        token_loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "train",
                loss,
                token_loss,
                logits,
                labels,
                auxiliary_loss,
            ),
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        token_loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "validation",
                loss,
                token_loss,
                logits,
                labels,
                auxiliary_loss,
            ),
            prog_bar=True,
        )

    def log_test_step(
        self,
        log_fn: Callable,
        loss: Tensor,
        token_loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "test",
                loss,
                token_loss,
                logits,
                labels,
                auxiliary_loss,
            ),
        )

    def _payload(
        self,
        stage: str,
        loss: Tensor,
        token_loss: Tensor,
        logits: Tensor,
        labels: Tensor,
        auxiliary_loss: Tensor | None,
    ) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {
            f"{stage}/loss": loss,
            f"{stage}/perplexity": self._perplexity.from_token_loss(token_loss),
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
