import math
from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from ._records import BertPretrainingStepOutput


class BertPretrainingMetricsLogger(nn.Module):
    def __init__(self):
        super().__init__()

    def log_training_step(
        self,
        log_fn: Callable,
        step_output: BertPretrainingStepOutput | Tensor,
        *legacy_outputs: Tensor | None,
    ) -> None:
        step_output = self._coerce_step_output(step_output, *legacy_outputs)
        log_fn(
            self._payload("train", step_output),
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        step_output: BertPretrainingStepOutput | Tensor,
        *legacy_outputs: Tensor | None,
    ) -> None:
        step_output = self._coerce_step_output(step_output, *legacy_outputs)
        log_fn(
            self._payload("validation", step_output),
            prog_bar=True,
        )

    def log_test_step(
        self,
        log_fn: Callable,
        step_output: BertPretrainingStepOutput | Tensor,
        *legacy_outputs: Tensor | None,
    ) -> None:
        step_output = self._coerce_step_output(step_output, *legacy_outputs)
        log_fn(
            self._payload("test", step_output),
        )

    def _coerce_step_output(
        self,
        step_output: BertPretrainingStepOutput | Tensor,
        *legacy_outputs: Tensor | None,
    ) -> BertPretrainingStepOutput:
        if isinstance(step_output, BertPretrainingStepOutput):
            return step_output
        if len(legacy_outputs) not in (6, 7):
            raise TypeError(
                "BertPretrainingMetricsLogger expected a "
                "BertPretrainingStepOutput or the legacy positional metrics."
            )
        auxiliary_loss = legacy_outputs[6] if len(legacy_outputs) == 7 else None
        return BertPretrainingStepOutput(
            total_loss=step_output,
            mlm_loss=legacy_outputs[0],
            nsp_loss=legacy_outputs[1],
            mlm_logits=legacy_outputs[2],
            mlm_labels=legacy_outputs[3],
            nsp_logits=legacy_outputs[4],
            next_sentence_labels=legacy_outputs[5],
            auxiliary_loss=auxiliary_loss,
        )

    def _payload(
        self,
        stage: str,
        step_output: BertPretrainingStepOutput,
    ) -> dict[str, Tensor | float]:
        payload: dict[str, Tensor | float] = {
            f"{stage}/loss": step_output.total_loss,
            f"{stage}/mlm/loss": step_output.mlm_loss,
            f"{stage}/mlm/perplexity": math.exp(step_output.mlm_loss.item()),
            f"{stage}/mlm/masked_accuracy": self._masked_accuracy(
                step_output.mlm_logits,
                step_output.mlm_labels,
            ),
            f"{stage}/mlm/masked_top_5_accuracy": self._masked_top_k_accuracy(
                step_output.mlm_logits,
                step_output.mlm_labels,
                k=5,
            ),
            f"{stage}/nsp/loss": step_output.nsp_loss,
            f"{stage}/nsp/accuracy": self._nsp_accuracy(
                step_output.nsp_logits,
                step_output.next_sentence_labels,
            ),
        }
        if step_output.auxiliary_loss is not None:
            payload[f"{stage}/auxiliary/loss"] = step_output.auxiliary_loss
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

    def _nsp_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        predictions = logits.argmax(dim=-1)
        return (predictions == labels).float().mean()
