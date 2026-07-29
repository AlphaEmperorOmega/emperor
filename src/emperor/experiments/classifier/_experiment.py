from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from emperor.experiments._auxiliary_loss import AuxiliaryLoss
from emperor.experiments._config_validation import _ExperimentConfigValidator

from ._metrics import ClassifierMetricsLogger
from ._records import ClassifierBatch, ClassifierStepOutput

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ClassifierExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        resolved_config = _ExperimentConfigValidator(
            "Classifier",
            minimum_output_dim=2,
        ).resolve(cfg)
        self.learning_rate = resolved_config.learning_rate
        self.num_classes = resolved_config.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = ClassifierMetricsLogger(self.num_classes)
        self._auxiliary_loss = AuxiliaryLoss("Classifier")

    def training_step(self, batch: ClassifierBatch, batch_idx: int) -> Tensor:
        output = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, output)
        return output.total_loss

    def on_train_epoch_start(self) -> None:
        self.metrics.reset_train_epoch()

    def on_train_epoch_end(self) -> None:
        self.metrics.log_train_epoch(self.log_dict)

    def validation_step(self, batch: ClassifierBatch, batch_idx: int) -> Tensor:
        inputs, _ = batch
        output = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, output, inputs)
        return output.total_loss

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset_validation_epoch()

    def on_validation_epoch_end(self) -> None:
        self.metrics.log_validation_epoch_and_gap(self.log_dict)
        if not self.trainer.sanity_checking:
            self.metrics.log_best_validation(self.log_dict, self.current_epoch)
            self.metrics.log_validation_examples(self.logger, self.current_epoch)

    def test_step(self, batch: ClassifierBatch, batch_idx: int) -> Tensor:
        output = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, output)
        return output.total_loss

    def on_before_optimizer_step(self, optimizer) -> None:
        health_metrics = self.optimizer_health_metrics(optimizer)
        if health_metrics:
            self.log_dict(
                health_metrics,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

    def _model_step(
        self,
        batch: ClassifierBatch,
    ) -> ClassifierStepOutput:
        inputs, labels = self._unpack_batch(batch)
        logits, resolved_auxiliary_loss = self._validate_model_output(
            self(inputs), labels
        )
        task_loss = self.loss_fn(logits, labels)
        loss = task_loss
        if resolved_auxiliary_loss is not None:
            loss = task_loss + resolved_auxiliary_loss
        return ClassifierStepOutput(loss, logits, labels)

    @staticmethod
    def _unpack_batch(batch: ClassifierBatch) -> ClassifierBatch:
        if len(batch) != 2:
            raise ValueError(
                "ClassifierExperiment batches must contain (inputs, labels)."
            )
        inputs, labels = batch
        if not isinstance(inputs, Tensor) or inputs.ndim < 1:
            raise ValueError("Classifier input must be a tensor with a batch dimension.")
        if not isinstance(labels, Tensor) or labels.ndim != 1:
            raise ValueError("Classifier labels must be a rank-1 tensor.")
        if inputs.size(0) != labels.size(0):
            raise ValueError(
                "Classifier inputs and labels must share the batch dimension."
            )
        return inputs, labels

    def _validate_model_output(
        self,
        output: object,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    "Classifier tuple outputs must be a two-item tuple containing "
                    "(logits, auxiliary_loss)."
                )
            logits, auxiliary_loss = output
        else:
            logits = output
            auxiliary_loss = None
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError(
                "Classifier logits must be a rank-2 tensor with shape "
                "[batch, classes]."
            )
        if logits.size(0) != labels.size(0):
            raise ValueError(
                "Classifier logits and labels must share the batch dimension."
            )
        if logits.size(1) != self.num_classes:
            raise ValueError(
                "Classifier logits class dimension must equal config.output_dim "
                f"({self.num_classes}), received {logits.size(1)}."
            )
        resolved_auxiliary_loss = None
        if auxiliary_loss is not None:
            resolved_auxiliary_loss = self._auxiliary_loss.resolve(
                auxiliary_loss,
                reference=logits,
            )
        return logits, resolved_auxiliary_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def optimizer_health_metrics(self, optimizer) -> dict[str, Tensor]:
        totals = self._optimizer_health_totals(optimizer)
        parameter_norm = totals["parameter_square_total"].sqrt()
        gradient_norm = totals["gradient_square_total"].sqrt()
        update_norm = totals["update_square_total"].sqrt()
        update_to_weight_ratio = torch.where(
            parameter_norm > 0,
            update_norm / parameter_norm.clamp_min(1e-12),
            torch.zeros_like(parameter_norm),
        )
        return {
            "gradients/global_norm": gradient_norm,
            "parameters/global_norm": parameter_norm,
            "updates/update_to_weight_ratio": update_to_weight_ratio,
            "gradients/nan_count": totals["gradient_nan_count"],
            "gradients/inf_count": totals["gradient_inf_count"],
        }

    def _optimizer_health_totals(self, optimizer) -> dict[str, Tensor]:
        totals = {
            "parameter_square_total": torch.tensor(0.0, device=self.device),
            "gradient_square_total": torch.tensor(0.0, device=self.device),
            "update_square_total": torch.tensor(0.0, device=self.device),
            "gradient_nan_count": torch.tensor(0.0, device=self.device),
            "gradient_inf_count": torch.tensor(0.0, device=self.device),
        }
        for group in optimizer.param_groups:
            learning_rate = float(group.get("lr", self.learning_rate))
            for parameter in group.get("params", []):
                self._accumulate_optimizer_parameter_health(
                    totals,
                    learning_rate,
                    parameter,
                )
        return totals

    def _accumulate_optimizer_parameter_health(
        self,
        totals: dict[str, Tensor],
        learning_rate: float,
        parameter,
    ) -> None:
        if parameter is None:
            return
        parameter_data = parameter.detach()
        totals["parameter_square_total"] = (
            totals["parameter_square_total"]
            + torch.nan_to_num(
                parameter_data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .pow(2)
            .sum()
        )

        gradient = parameter.grad
        if gradient is None:
            return
        gradient_data = gradient.detach()
        finite_gradient = torch.nan_to_num(
            gradient_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        totals["gradient_square_total"] = (
            totals["gradient_square_total"] + finite_gradient.pow(2).sum()
        )
        totals["update_square_total"] = (
            totals["update_square_total"]
            + (finite_gradient * learning_rate).pow(2).sum()
        )
        totals["gradient_nan_count"] = (
            totals["gradient_nan_count"] + torch.isnan(gradient_data).sum()
        )
        totals["gradient_inf_count"] = (
            totals["gradient_inf_count"] + torch.isinf(gradient_data).sum()
        )
