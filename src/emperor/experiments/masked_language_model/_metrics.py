from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from emperor.experiments._perplexity import Perplexity

from ._records import MaskedLanguageModelStepOutput


class MaskedLanguageModelMetricsLogger(nn.Module):
    def __init__(self):
        super().__init__()
        self._perplexity = Perplexity()

    def log_training_step(
        self,
        log_fn: Callable,
        output: MaskedLanguageModelStepOutput,
    ) -> None:
        log_fn(
            self._payload("train", output),
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        output: MaskedLanguageModelStepOutput,
    ) -> None:
        log_fn(
            self._payload("validation", output),
            prog_bar=True,
        )

    def log_test_step(
        self,
        log_fn: Callable,
        output: MaskedLanguageModelStepOutput,
    ) -> None:
        log_fn(self._payload("test", output))

    def _payload(
        self,
        stage: str,
        output: MaskedLanguageModelStepOutput,
    ) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {
            f"{stage}/loss": output.total_loss,
            f"{stage}/perplexity": self._perplexity.from_token_loss(
                output.cross_entropy
            ),
            f"{stage}/masked/accuracy": self._masked_accuracy(
                output.logits,
                output.labels,
            ),
            f"{stage}/masked/top_5_accuracy": self._masked_top_k_accuracy(
                output.logits,
                output.labels,
                k=5,
            ),
        }
        if output.auxiliary_loss is not None:
            payload[f"{stage}/auxiliary/loss"] = output.auxiliary_loss
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
