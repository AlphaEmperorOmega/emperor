from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from emperor.experiments._perplexity import Perplexity

from ._records import LanguageModelStepOutput


class LanguageModelMetricsLogger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._perplexity = Perplexity()

    def log_training_step(
        self,
        log_fn: Callable,
        output: LanguageModelStepOutput | Tensor,
    ) -> None:
        log_fn(self._payload("train", output), prog_bar=True)

    def log_validation_step(
        self,
        log_fn: Callable,
        output: LanguageModelStepOutput | Tensor,
    ) -> None:
        log_fn(self._payload("validation", output), prog_bar=True)

    def log_test_step(
        self,
        log_fn: Callable,
        output: LanguageModelStepOutput | Tensor,
    ) -> None:
        log_fn(self._payload("test", output))

    def _payload(
        self,
        stage: str,
        output: LanguageModelStepOutput | Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(output, LanguageModelStepOutput):
            total_loss = output.total_loss
            cross_entropy = output.cross_entropy
            auxiliary_loss = output.auxiliary_loss
        else:
            total_loss = output
            cross_entropy = output
            auxiliary_loss = output.new_zeros(())
        perplexity = self._perplexity.from_token_loss(cross_entropy)
        return {
            f"{stage}/loss": total_loss,
            f"{stage}/cross_entropy": cross_entropy,
            f"{stage}/perplexity": perplexity,
            f"{stage}/auxiliary_loss": auxiliary_loss,
        }
