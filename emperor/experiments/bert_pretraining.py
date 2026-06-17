import math
import torch
import torch.nn as nn

from torch import Tensor
from lightning import LightningModule
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


BertPretrainingBatch = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
BertPretrainingStepOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
]


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
        outputs = self._model_step_outputs(batch)
        self.metrics.log_training_step(self.log_dict, *outputs)
        return outputs[0]

    def validation_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        outputs = self._model_step_outputs(batch)
        self.metrics.log_validation_step(self.log_dict, *outputs)
        return outputs[0]

    def test_step(self, batch: BertPretrainingBatch, batch_idx: int) -> Tensor:
        outputs = self._model_step_outputs(batch)
        self.metrics.log_test_step(self.log_dict, *outputs)
        return outputs[0]

    def _model_step(self, batch: BertPretrainingBatch) -> Tensor:
        total_loss, *_ = self._model_step_outputs(batch)
        return total_loss

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
        return (
            total_loss,
            mlm_loss,
            nsp_loss,
            mlm_logits,
            mlm_labels,
            nsp_logits,
            next_sentence_labels,
            auxiliary_loss,
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
        total_loss: Tensor,
        mlm_loss: Tensor,
        nsp_loss: Tensor,
        mlm_logits: Tensor,
        mlm_labels: Tensor,
        nsp_logits: Tensor,
        next_sentence_labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "train",
                total_loss,
                mlm_loss,
                nsp_loss,
                mlm_logits,
                mlm_labels,
                nsp_logits,
                next_sentence_labels,
                auxiliary_loss,
            ),
            prog_bar=True,
        )

    def log_validation_step(
        self,
        log_fn: Callable,
        total_loss: Tensor,
        mlm_loss: Tensor,
        nsp_loss: Tensor,
        mlm_logits: Tensor,
        mlm_labels: Tensor,
        nsp_logits: Tensor,
        next_sentence_labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "validation",
                total_loss,
                mlm_loss,
                nsp_loss,
                mlm_logits,
                mlm_labels,
                nsp_logits,
                next_sentence_labels,
                auxiliary_loss,
            ),
            prog_bar=True,
        )

    def log_test_step(
        self,
        log_fn: Callable,
        total_loss: Tensor,
        mlm_loss: Tensor,
        nsp_loss: Tensor,
        mlm_logits: Tensor,
        mlm_labels: Tensor,
        nsp_logits: Tensor,
        next_sentence_labels: Tensor,
        auxiliary_loss: Tensor | None = None,
    ) -> None:
        log_fn(
            self._payload(
                "test",
                total_loss,
                mlm_loss,
                nsp_loss,
                mlm_logits,
                mlm_labels,
                nsp_logits,
                next_sentence_labels,
                auxiliary_loss,
            ),
        )

    def _payload(
        self,
        stage: str,
        total_loss: Tensor,
        mlm_loss: Tensor,
        nsp_loss: Tensor,
        mlm_logits: Tensor,
        mlm_labels: Tensor,
        nsp_logits: Tensor,
        next_sentence_labels: Tensor,
        auxiliary_loss: Tensor | None,
    ) -> dict[str, Tensor | float]:
        payload: dict[str, Tensor | float] = {
            f"{stage}/loss": total_loss,
            f"{stage}/mlm/loss": mlm_loss,
            f"{stage}/mlm/perplexity": math.exp(mlm_loss.item()),
            f"{stage}/mlm/masked_accuracy": self._masked_accuracy(
                mlm_logits,
                mlm_labels,
            ),
            f"{stage}/mlm/masked_top_5_accuracy": self._masked_top_k_accuracy(
                mlm_logits,
                mlm_labels,
                k=5,
            ),
            f"{stage}/nsp/loss": nsp_loss,
            f"{stage}/nsp/accuracy": self._nsp_accuracy(
                nsp_logits,
                next_sentence_labels,
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

    def _nsp_accuracy(self, logits: Tensor, labels: Tensor) -> Tensor:
        predictions = logits.argmax(dim=-1)
        return (predictions == labels).float().mean()
