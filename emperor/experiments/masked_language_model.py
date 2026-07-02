import math
import torch
import torch.nn as nn

from torch import Tensor
from lightning import LightningModule
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from emperor.config import ModelConfig


MaskedLanguageModelBatch = (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
)
MaskedLanguageModelStepOutput = tuple[Tensor, Tensor, Tensor, Tensor | None]


class MaskedLanguageModelExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = MaskedLanguageModelMetricsLogger()

    def training_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_training_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def validation_step(
        self, batch: MaskedLanguageModelBatch, batch_idx: int
    ) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_validation_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def test_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, logits, labels, auxiliary_loss = self._model_step_outputs(batch)
        self.metrics.log_test_step(
            self.log_dict,
            loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def _model_step(self, batch: MaskedLanguageModelBatch) -> Tensor:
        loss, _, _, _ = self._model_step_outputs(batch)
        return loss

    def _model_step_outputs(
        self, batch: MaskedLanguageModelBatch
    ) -> MaskedLanguageModelStepOutput:
        input_ids, labels, attention_mask, token_type_ids = self._unpack_batch(batch)
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        forward_kwargs = {}
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask.to(self.device)
        if token_type_ids is not None:
            forward_kwargs["token_type_ids"] = token_type_ids.to(self.device)

        output = self(input_ids, **forward_kwargs)
        auxiliary_loss = None
        if isinstance(output, tuple):
            logits, auxiliary_loss = output[0], output[-1]
        else:
            logits = output

        loss = self.loss_fn(logits.transpose(1, 2), labels)
        if auxiliary_loss is not None and auxiliary_loss.item() != 0.0:
            loss = loss + auxiliary_loss
        return loss, logits, labels, auxiliary_loss

    def _unpack_batch(
        self, batch: MaskedLanguageModelBatch
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if len(batch) == 2:
            input_ids, labels = batch
            return input_ids, labels, None, None
        if len(batch) == 3:
            input_ids, labels, attention_mask = batch
            return input_ids, labels, attention_mask, None
        if len(batch) == 4:
            input_ids, labels, attention_mask, token_type_ids = batch
            return input_ids, labels, attention_mask, token_type_ids
        raise ValueError(
            "MaskedLanguageModelExperiment batches must contain "
            "(input_ids, labels), (input_ids, labels, attention_mask), "
            "or (input_ids, labels, attention_mask, token_type_ids)."
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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
