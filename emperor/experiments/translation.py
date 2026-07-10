from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torchmetrics.text import SacreBLEUScore

from emperor.datasets.text.translation.multi30k import PAD_ID

if TYPE_CHECKING:
    from emperor.config import ModelConfig


TranslationBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class TranslationStepOutput:
    total_loss: Tensor
    nll: Tensor
    logits: Tensor
    labels: Tensor
    auxiliary_loss: Tensor


class TranslationExperiment(LightningModule):
    """Shared teacher-forced training and corpus evaluation for translation models."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        experiment_config = cfg.experiment_config
        self.learning_rate = float(cfg.learning_rate)
        self.vocab_size = int(cfg.output_dim)
        self.model_dim = int(cfg.hidden_dim)
        self.pad_token_id = int(getattr(experiment_config, "pad_token_id", PAD_ID))
        self.label_smoothing = float(getattr(experiment_config, "label_smoothing", 0.1))
        self.warmup_steps = int(getattr(experiment_config, "warmup_steps", 4_000))
        self.generation_metrics_flag = bool(
            getattr(experiment_config, "generation_metrics_flag", True)
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,
            label_smoothing=self.label_smoothing,
        )
        self.nll_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self._validation_predictions: list[str] = []
        self._validation_references: list[str] = []
        self._test_predictions: list[str] = []
        self._test_references: list[str] = []

    def training_step(self, batch: TranslationBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self._log_step("train", output, prog_bar=True)
        return output.total_loss

    def validation_step(self, batch: TranslationBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self._log_step("validation", output, prog_bar=True)
        self._collect_generation("validation", batch)
        return output.total_loss

    def test_step(self, batch: TranslationBatch, batch_idx: int) -> Tensor:
        output = self._model_step_outputs(batch)
        self._log_step("test", output, prog_bar=False)
        self._collect_generation("test", batch)
        return output.total_loss

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions.clear()
        self._validation_references.clear()

    def on_validation_epoch_end(self) -> None:
        self._log_corpus_bleu(
            "validation",
            self._validation_predictions,
            self._validation_references,
        )

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()
        self._test_references.clear()

    def on_test_epoch_end(self) -> None:
        self._log_corpus_bleu(
            "test",
            self._test_predictions,
            self._test_references,
        )

    def _model_step(self, batch: TranslationBatch) -> Tensor:
        return self._model_step_outputs(batch).total_loss

    def _model_step_outputs(self, batch: TranslationBatch) -> TranslationStepOutput:
        source_ids, target_ids = self._unpack_batch(batch)
        source_ids = source_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        target_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        logits, auxiliary_loss = self(source_ids, target_input_ids)
        if auxiliary_loss is None:
            auxiliary_loss = logits.new_zeros(())
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        smoothed_loss = self.loss_fn(flat_logits, flat_labels)
        nll = self.nll_fn(flat_logits, flat_labels)
        total_loss = smoothed_loss + auxiliary_loss
        return TranslationStepOutput(
            total_loss=total_loss,
            nll=nll,
            logits=logits,
            labels=labels,
            auxiliary_loss=auxiliary_loss,
        )

    def _unpack_batch(self, batch: TranslationBatch) -> TranslationBatch:
        if len(batch) != 2:
            raise ValueError(
                "TranslationExperiment batches must contain (source_ids, target_ids)."
            )
        source_ids, target_ids = batch
        if source_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError(
                "Translation source and target IDs must be rank-2 tensors."
            )
        if target_ids.size(1) < 2:
            raise ValueError(
                "Translation target sequences must contain at least 2 IDs."
            )
        return source_ids, target_ids

    def _log_step(
        self,
        stage: str,
        output: TranslationStepOutput,
        *,
        prog_bar: bool,
    ) -> None:
        valid_tokens = output.labels != self.pad_token_id
        predictions = output.logits.argmax(dim=-1)
        if bool(valid_tokens.any().item()):
            token_accuracy = (
                (predictions[valid_tokens] == output.labels[valid_tokens])
                .float()
                .mean()
            )
        else:
            token_accuracy = output.logits.new_zeros(())
        perplexity = torch.exp(output.nll.detach().clamp(max=math.log(1e9)))
        self.log_dict(
            {
                f"{stage}/loss": output.total_loss,
                f"{stage}/nll": output.nll,
                f"{stage}/perplexity": perplexity,
                f"{stage}/token_accuracy": token_accuracy,
                f"{stage}/auxiliary_loss": output.auxiliary_loss,
            },
            prog_bar=prog_bar,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=output.labels.size(0),
        )

    def _collect_generation(
        self,
        stage: str,
        batch: TranslationBatch,
    ) -> None:
        if not self.generation_metrics_flag:
            return
        datamodule = self._translation_datamodule()
        if datamodule is None or not hasattr(datamodule, "decode_batch"):
            return
        source_ids, target_ids = batch
        generated_ids = self.generate(
            source_ids.to(self.device),
            max_length=target_ids.size(1),
        )
        predictions = datamodule.decode_batch(generated_ids.detach().cpu())
        references = datamodule.decode_batch(target_ids.detach().cpu())
        if stage == "validation":
            self._validation_predictions.extend(predictions)
            self._validation_references.extend(references)
        else:
            self._test_predictions.extend(predictions)
            self._test_references.extend(references)

    def _translation_datamodule(self):
        trainer = getattr(self, "_trainer", None)
        return getattr(trainer, "datamodule", None) if trainer is not None else None

    def _log_corpus_bleu(
        self,
        stage: str,
        predictions: list[str],
        references: list[str],
    ) -> None:
        if not self.generation_metrics_flag or not predictions:
            return
        metric = SacreBLEUScore(tokenize="13a").to(self.device)
        bleu = metric(predictions, [[reference] for reference in references])
        self.log(f"{stage}/bleu", bleu, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        def inverse_square_root_schedule(step: int) -> float:
            safe_step = max(1, step + 1)
            return self.model_dim**-0.5 * min(
                safe_step**-0.5,
                safe_step * self.warmup_steps**-1.5,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=inverse_square_root_schedule,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


__all__ = [
    "TranslationBatch",
    "TranslationExperiment",
    "TranslationStepOutput",
]
