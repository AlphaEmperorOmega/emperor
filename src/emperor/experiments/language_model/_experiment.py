from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from ._metrics import LanguageModelMetricsLogger
from ._records import LanguageModelBatch, LanguageModelStepOutput

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LanguageModelExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.vocab_size = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = LanguageModelMetricsLogger()

    def training_step(self, batch: LanguageModelBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self.metrics.log_training_step(self.log_dict, output)
        return output.total_loss

    def validation_step(self, batch: LanguageModelBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self.metrics.log_validation_step(self.log_dict, output)
        return output.total_loss

    def test_step(self, batch: LanguageModelBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self.metrics.log_test_step(self.log_dict, output)
        return output.total_loss

    def _model_step(self, batch: LanguageModelBatch) -> Tensor:
        return self._model_step_outputs(batch).total_loss

    def _model_step_outputs(
        self,
        batch: LanguageModelBatch,
    ) -> LanguageModelStepOutput:
        tokens, targets = self._unpack_batch(batch)
        tokens = tokens.to(self.device)
        targets = targets.to(self.device)
        output = self(tokens)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Language-model tuple outputs must contain "
                    "(logits, auxiliary_loss)."
                )
            logits, auxiliary_loss = output
        else:
            logits = output
            auxiliary_loss = None
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(
                "Language-model logits must be a rank-3 tensor with shape "
                "[batch, sequence, vocabulary]."
            )
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Language-model logits and labels must share batch and sequence "
                f"dimensions, received {tuple(logits.shape)} and "
                f"{tuple(targets.shape)}."
            )
        if logits.size(-1) != self.vocab_size:
            raise ValueError(
                "Language-model logits vocabulary dimension must equal "
                f"config.output_dim ({self.vocab_size}), received "
                f"{logits.size(-1)}."
            )
        if auxiliary_loss is None:
            auxiliary_loss = logits.new_zeros(())
        elif not isinstance(auxiliary_loss, Tensor) or auxiliary_loss.numel() != 1:
            raise ValueError("Language-model auxiliary loss must be a scalar tensor.")
        else:
            auxiliary_loss = auxiliary_loss.reshape(())
        cross_entropy = self.loss_fn(logits.transpose(1, 2), targets)
        return LanguageModelStepOutput(
            total_loss=cross_entropy + auxiliary_loss,
            cross_entropy=cross_entropy,
            logits=logits,
            labels=targets,
            auxiliary_loss=auxiliary_loss,
        )

    @staticmethod
    def _unpack_batch(batch: LanguageModelBatch) -> LanguageModelBatch:
        if len(batch) != 2:
            raise ValueError(
                "LanguageModelExperiment batches must contain (input_ids, labels)."
            )
        tokens, targets = batch
        if tokens.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                "Language-model input IDs and labels must be rank-2 tensors."
            )
        if tokens.shape != targets.shape:
            raise ValueError(
                "Language-model input IDs and labels must have equal shapes, "
                f"received {tuple(tokens.shape)} and {tuple(targets.shape)}."
            )
        return tokens, targets

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
