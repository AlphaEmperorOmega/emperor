from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import Callback

from emperor.linears._monitoring.capture import _LinearCaptureLifecycle
from emperor.linears._monitoring.emitter import _LinearMetricEmitter

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from torch.optim import Optimizer


class LinearMonitorCallback(Callback):
    """Log activations, parameters, gradients, and matrix health for linear layers."""

    DEAD_FEATURE_RELATIVE_FLOOR = _LinearMetricEmitter.DEAD_FEATURE_RELATIVE_FLOOR

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_weight_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(log_every_n_steps, bool) or not isinstance(
            log_every_n_steps, int
        ):
            raise TypeError("log_every_n_steps must be an int.")
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        if not isinstance(log_weight_conditioning, bool):
            raise TypeError("log_weight_conditioning must be a bool.")

        self._emitter = _LinearMetricEmitter(self.DEAD_FEATURE_RELATIVE_FLOOR)
        self._capture = _LinearCaptureLifecycle(
            log_every_n_steps,
            log_weight_conditioning,
            self._emitter,
        )

    @property
    def log_every_n_steps(self) -> int:
        return self._capture.log_every_n_steps

    @log_every_n_steps.setter
    def log_every_n_steps(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("log_every_n_steps must be an int.")
        if value <= 0:
            raise ValueError("log_every_n_steps must be greater than 0.")
        self._capture.log_every_n_steps = value

    @property
    def log_weight_conditioning(self) -> bool:
        return self._capture.log_weight_conditioning

    @log_weight_conditioning.setter
    def log_weight_conditioning(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("log_weight_conditioning must be a bool.")
        self._capture.log_weight_conditioning = value

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._capture.begin_fit(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        self._capture.begin_train_batch(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        self._capture.finish_train_batch(trainer, pl_module)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
    ) -> None:
        self._capture.before_optimizer_step(trainer, pl_module, optimizer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._capture.close()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self._capture.close()
