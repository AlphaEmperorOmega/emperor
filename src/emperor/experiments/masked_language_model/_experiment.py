from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from emperor.experiments._auxiliary_loss import AuxiliaryLoss

from ._metrics import MaskedLanguageModelMetricsLogger

if TYPE_CHECKING:
    from emperor.config import ModelConfig


MaskedLanguageModelBatch = (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
)
MaskedLanguageModelStepOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]


class MaskedLanguageModelExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics = MaskedLanguageModelMetricsLogger()
        self._auxiliary_loss = AuxiliaryLoss("Masked-language-model")

    def training_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, token_loss, logits, labels, auxiliary_loss = self._model_step_outputs(
            batch
        )
        self.metrics.log_training_step(
            self.log_dict,
            loss,
            token_loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def validation_step(
        self, batch: MaskedLanguageModelBatch, batch_idx: int
    ) -> Tensor:
        loss, token_loss, logits, labels, auxiliary_loss = self._model_step_outputs(
            batch
        )
        self.metrics.log_validation_step(
            self.log_dict,
            loss,
            token_loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def test_step(self, batch: MaskedLanguageModelBatch, batch_idx: int) -> Tensor:
        loss, token_loss, logits, labels, auxiliary_loss = self._model_step_outputs(
            batch
        )
        self.metrics.log_test_step(
            self.log_dict,
            loss,
            token_loss,
            logits,
            labels,
            auxiliary_loss,
        )
        return loss

    def _model_step(self, batch: MaskedLanguageModelBatch) -> Tensor:
        loss, _, _, _, _ = self._model_step_outputs(batch)
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

        logits, auxiliary_loss, resolved_auxiliary_loss = self._validate_model_output(
            self(input_ids, **forward_kwargs),
            labels,
        )

        task_loss = self.loss_fn(logits.transpose(1, 2), labels)
        loss = task_loss
        if resolved_auxiliary_loss is not None:
            loss = task_loss + resolved_auxiliary_loss
        return loss, task_loss, logits, labels, auxiliary_loss

    def _unpack_batch(
        self, batch: MaskedLanguageModelBatch
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if len(batch) == 2:
            input_ids, labels = batch
            unpacked_batch = input_ids, labels, None, None
        elif len(batch) == 3:
            input_ids, labels, attention_mask = batch
            unpacked_batch = input_ids, labels, attention_mask, None
        elif len(batch) == 4:
            input_ids, labels, attention_mask, token_type_ids = batch
            unpacked_batch = input_ids, labels, attention_mask, token_type_ids
        else:
            raise ValueError(
                "MaskedLanguageModelExperiment batches must contain "
                "(input_ids, labels), (input_ids, labels, attention_mask), "
                "or (input_ids, labels, attention_mask, token_type_ids)."
            )
        self._validate_batch_geometry(*unpacked_batch)
        return unpacked_batch

    @staticmethod
    def _validate_batch_geometry(
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None,
    ) -> None:
        if (
            not isinstance(input_ids, Tensor)
            or not isinstance(labels, Tensor)
            or input_ids.ndim != 2
            or labels.ndim != 2
        ):
            raise ValueError(
                "Masked-language-model input IDs and labels must be rank-2 tensors."
            )
        if input_ids.shape != labels.shape:
            raise ValueError(
                "Masked-language-model input IDs and labels must have equal shapes."
            )
        if attention_mask is not None and (
            not isinstance(attention_mask, Tensor)
            or attention_mask.ndim != 2
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Masked-language-model attention mask must be a rank-2 tensor "
                "matching the input shape."
            )
        if token_type_ids is not None and (
            not isinstance(token_type_ids, Tensor)
            or token_type_ids.ndim != 2
            or token_type_ids.shape != input_ids.shape
        ):
            raise ValueError(
                "Masked-language-model token-type IDs must be a rank-2 tensor "
                "matching the input shape."
            )

    def _validate_model_output(
        self,
        output: object,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Masked-language-model tuple outputs must be a two-item tuple "
                    "containing (logits, auxiliary_loss)."
                )
            logits, auxiliary_loss = output
        else:
            logits = output
            auxiliary_loss = None
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(
                "Masked-language-model MLM logits must be a rank-3 tensor with "
                "shape [batch, sequence, vocabulary]."
            )
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "Masked-language-model logits and labels must share batch and "
                "sequence dimensions."
            )
        if logits.size(-1) != self.vocab_size:
            raise ValueError(
                "Masked-language-model logits vocabulary dimension must equal "
                f"config.output_dim ({self.vocab_size}), received "
                f"{logits.size(-1)}."
            )
        resolved_auxiliary_loss = None
        if auxiliary_loss is not None:
            resolved_auxiliary_loss = self._auxiliary_loss.resolve(
                auxiliary_loss,
                reference=logits,
            )
        return logits, auxiliary_loss, resolved_auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
