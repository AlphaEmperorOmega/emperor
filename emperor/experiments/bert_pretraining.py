import math
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from lightning import LightningModule
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from emperor.config import ModelConfig


BertPretrainingBatch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class BertPretrainingStepOutput:
    total_loss: Tensor
    mlm_loss: Tensor
    nsp_loss: Tensor
    mlm_logits: Tensor
    mlm_labels: Tensor
    nsp_logits: Tensor
    next_sentence_labels: Tensor
    auxiliary_loss: Tensor | None = None


class BertPretrainingExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        self.metrics = BertPretrainingMetricsLogger()

    def training_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_training_step(self.log_dict, step_output)
        return step_output.total_loss

    def validation_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_validation_step(self.log_dict, step_output)
        return step_output.total_loss

    def test_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        step_output = self._model_step_outputs(batch)
        self.metrics.log_test_step(self.log_dict, step_output)
        return step_output.total_loss

    def _model_step(self, batch: BertPretrainingBatch) -> Tensor:
        return self._model_step_outputs(batch).total_loss

    def _model_step_outputs(
        self,
        batch: BertPretrainingBatch,
    ) -> BertPretrainingStepOutput:
        (
            input_ids,
            mlm_labels,
            attention_mask,
            token_type_ids,
            next_sentence_labels,
        ) = self._unpack_batch(batch)
        input_ids = input_ids.to(self.device)
        mlm_labels = mlm_labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        next_sentence_labels = next_sentence_labels.to(self.device)

        mlm_logits, nsp_logits, auxiliary_loss = self(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        mlm_loss = self.mlm_loss_fn(mlm_logits.transpose(1, 2), mlm_labels)
        nsp_loss = self.nsp_loss_fn(nsp_logits, next_sentence_labels)
        total_loss = mlm_loss + nsp_loss
        if auxiliary_loss is not None and bool(
            torch.any(auxiliary_loss.detach() != 0.0).item()
        ):
            total_loss = total_loss + auxiliary_loss
        return BertPretrainingStepOutput(
            total_loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
            mlm_logits=mlm_logits,
            mlm_labels=mlm_labels,
            nsp_logits=nsp_logits,
            next_sentence_labels=next_sentence_labels,
            auxiliary_loss=auxiliary_loss,
        )

    def _unpack_batch(self, batch: BertPretrainingBatch) -> BertPretrainingBatch:
        if len(batch) != 5:
            raise ValueError(
                "BertPretrainingExperiment batches must contain "
                "(input_ids, mlm_labels, attention_mask, token_type_ids, "
                "next_sentence_labels)."
            )
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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
