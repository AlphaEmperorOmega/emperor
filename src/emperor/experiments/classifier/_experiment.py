from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from ._metrics import ClassifierMetricsLogger

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ClassifierExperiment(LightningModule):
    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = self.cfg.learning_rate
        self.num_classes = self.cfg.output_dim
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = ClassifierMetricsLogger(self.num_classes)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_training_step(self.log_dict, loss, logits, Y)
        return loss

    def on_train_epoch_start(self) -> None:
        self.metrics.reset_train_epoch()

    def on_train_epoch_end(self) -> None:
        self.metrics.log_train_epoch(self.log_dict)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        X, _ = batch
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_validation_step(self.log_dict, loss, logits, Y, X)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset_validation_epoch()

    def on_validation_epoch_end(self) -> None:
        self.metrics.log_validation_epoch_and_gap(self.log_dict)
        if not self.trainer.sanity_checking:
            self.metrics.log_best_validation(self.log_dict, self.current_epoch)
            self.metrics.log_validation_examples(self.logger, self.current_epoch)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, logits, Y = self._model_step(batch)
        self.metrics.log_test_step(self.log_dict, loss, logits, Y)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        health_metrics = self.optimizer_health_metrics(optimizer)
        if health_metrics:
            self.log_dict(
                health_metrics,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

    def _model_step(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        X, Y = batch
        output = self(X)
        if isinstance(output, tuple):
            logits, auxiliary_loss = output[0], output[-1]
            loss = self.loss_fn(logits, Y)
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss
        else:
            logits = output
            loss = self.loss_fn(logits, Y)
        return loss, logits, Y

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
