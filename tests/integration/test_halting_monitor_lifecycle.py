from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from emperor.halting import (
    HaltingHiddenStateModeOptions,
    HaltingMonitorCallback,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig

SCALAR_SUFFIXES = (
    "depth/ponder_cost_mean",
    "depth/ponder_cost_std",
    "depth/step_count",
    "halt/halted_fraction",
    "halt/accumulated_halt_prob_mean",
    "halt/remaining_mass_mean",
    "halt/saturation_fraction",
    "loss/ponder_loss",
)


def layer_config(model_config: LinearLayerConfig) -> LayerConfig:
    return LayerConfig(
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=model_config,
    )


def stack_config(
    dim: int,
    output_dim: int,
    *,
    bias_flag: bool,
    bias_option: LastLayerBiasOptions,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=bias_option,
        apply_output_pipeline_flag=False,
        layer_config=layer_config(LinearLayerConfig(bias_flag=bias_flag)),
    )


def recurrent_config(
    *,
    dim: int = 2,
    max_steps: int = 3,
    threshold: float = 1.0,
) -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=dim,
        output_dim=dim,
        max_steps=max_steps,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=stack_config(
            dim,
            dim,
            bias_flag=True,
            bias_option=LastLayerBiasOptions.DEFAULT,
        ),
        gate_config=None,
        residual_config=None,
        halting_config=StickBreakingConfig(
            input_dim=dim,
            threshold=threshold,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=stack_config(
                dim,
                2,
                bias_flag=False,
                bias_option=LastLayerBiasOptions.DISABLED,
            ),
        ),
        memory_config=None,
    )


class _HaltingTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        max_steps: int = 3,
        threshold: float = 1.0,
        fail_after_forward: bool = False,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.recurrent = recurrent_config(
            max_steps=max_steps,
            threshold=threshold,
        ).build()
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.logged_calls: list[tuple[int, str]] = []

    def training_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        (inputs,) = batch
        state = self.recurrent(LayerState(hidden=inputs))
        if self.fail_after_forward:
            raise RuntimeError("deliberate halting lifecycle failure")
        loss = state.hidden.square().mean()
        if state.loss is not None:
            loss = loss + state.loss
        return loss

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.logged_calls.append((int(self.global_step), name))
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class _IdleHaltingTrainingModule(_HaltingTrainingModule):
    def __init__(self) -> None:
        super().__init__(learning_rate=0.0)
        self.idle_parameter = torch.nn.Parameter(torch.zeros(()))

    def training_step(
        self,
        batch: tuple[Tensor],
        batch_idx: int,
    ) -> Tensor:
        return self.idle_parameter.square()


class _HistoryObserver(Callback):
    def __init__(self, monitor: HaltingMonitorCallback) -> None:
        super().__init__()
        self.monitor = monitor
        self.lengths: list[int] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        history = self.monitor._survival_history["recurrent.halting_model"]
        self.lengths.append(len(history))


def trainer(
    root: Path,
    callbacks: list[Callback],
    *,
    logger: bool | TensorBoardLogger = False,
) -> Trainer:
    return Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=root,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def loader(num_batches: int = 1) -> DataLoader:
    values = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 1.5],
            [-1.0, 0.25],
            [2.0, -0.5],
            [0.75, 1.25],
            [-0.25, -1.5],
        ]
    )
    return DataLoader(
        TensorDataset(values[: num_batches * 2]),
        batch_size=2,
        shuffle=False,
    )


class HaltingMonitorLifecycleTests(unittest.TestCase):
    def test_real_trainer_logs_exact_survival_metrics_and_updates_gate(self) -> None:
        torch.manual_seed(7)
        model = _HaltingTrainingModule(max_steps=3, threshold=1.0)
        output_weight = model.recurrent.halting_model.halting_gate_model[
            -1
        ].model.weight_params
        initial_weight = output_weight.detach().clone()
        monitor = HaltingMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [monitor])
            fit_trainer.fit(model, train_dataloaders=loader())

        prefix = "recurrent.halting_model"
        expected_names = {f"{prefix}/{suffix}" for suffix in SCALAR_SUFFIXES}
        self.assertEqual(set(fit_trainer.logged_metrics), expected_names)
        self.assertEqual({name for _, name in model.logged_calls}, expected_names)
        for metric_name, metric in fit_trainer.logged_metrics.items():
            self.assertTrue(torch.isfinite(metric), metric_name)
        torch.testing.assert_close(
            fit_trainer.logged_metrics[f"{prefix}/depth/step_count"],
            torch.tensor(3.0),
        )
        torch.testing.assert_close(
            fit_trainer.logged_metrics[f"{prefix}/halt/halted_fraction"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            fit_trainer.logged_metrics[f"{prefix}/halt/saturation_fraction"],
            torch.tensor(1.0),
        )
        self.assertFalse(torch.equal(output_weight.detach(), initial_weight))
        self.assertNotIn("_usage_tracker", model.recurrent.halting_model._modules)
        self.assertNotIn(
            "update_halting_state",
            model.recurrent.halting_model.__dict__,
        )
        self.assertNotIn(
            "finalize_weighted_accumulation",
            model.recurrent.halting_model.__dict__,
        )
        self.assertEqual(monitor._halting_layers, [])
        self.assertEqual(monitor._survival_history, {})

    def test_real_trainer_uses_global_step_cadence_and_bounded_history(self) -> None:
        torch.manual_seed(11)
        model = _HaltingTrainingModule(learning_rate=0.0)
        monitor = HaltingMonitorCallback(log_every_n_steps=2, history_size=2)
        observer = _HistoryObserver(monitor)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(
                Path(temporary_directory),
                [monitor, observer],
                logger=TensorBoardLogger(
                    save_dir=temporary_directory,
                    name="cadence",
                ),
            )
            fit_trainer.fit(model, train_dataloaders=loader(num_batches=3))

        self.assertEqual({step for step, _ in model.logged_calls}, {2})
        self.assertEqual(len(model.logged_calls), len(SCALAR_SUFFIXES))
        self.assertEqual(observer.lengths, [0, 1, 1])

    def test_real_tensorboard_logger_receives_histograms_and_heatmap(self) -> None:
        torch.manual_seed(13)
        model = _HaltingTrainingModule(max_steps=1, learning_rate=0.0)
        monitor = HaltingMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="halting",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [monitor],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader())
            tensorboard_logger.experiment.flush()

            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()
            tags = events.Tags()

        prefix = "recurrent.halting_model"
        survival_tag = f"{prefix}/histogram/survival"
        ponder_tag = f"{prefix}/histogram/ponder_cost"
        heatmap_tag = f"{prefix}/heatmap/survival"
        self.assertIn(survival_tag, tags["histograms"])
        self.assertIn(ponder_tag, tags["histograms"])
        self.assertIn(heatmap_tag, tags["images"])
        self.assertEqual(events.Histograms(survival_tag)[0].step, 1)
        self.assertEqual(events.Histograms(ponder_tag)[0].step, 1)
        self.assertEqual(events.Images(heatmap_tag)[0].step, 1)

    def test_real_tensorboard_logger_skips_empty_histograms(self) -> None:
        model = _IdleHaltingTrainingModule()
        monitor = HaltingMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="empty-halting",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [monitor],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader())
            tensorboard_logger.experiment.flush()

            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()

        self.assertEqual(events.Tags()["histograms"], [])
        prefix = "recurrent.halting_model"
        self.assertEqual(
            {name.removeprefix(f"{prefix}/") for name in fit_trainer.logged_metrics},
            set(SCALAR_SUFFIXES),
        )
        for metric in fit_trainer.logged_metrics.values():
            torch.testing.assert_close(metric, torch.zeros_like(metric))

    def test_real_trainer_exception_invokes_monitor_cleanup(self) -> None:
        model = _HaltingTrainingModule(
            fail_after_forward=True,
            learning_rate=0.0,
        )
        monitor = HaltingMonitorCallback(log_every_n_steps=1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [monitor])
            with self.assertRaisesRegex(
                RuntimeError,
                "^deliberate halting lifecycle failure$",
            ):
                fit_trainer.fit(model, train_dataloaders=loader())

        halting_model = model.recurrent.halting_model
        self.assertNotIn("_usage_tracker", halting_model._modules)
        self.assertNotIn("update_halting_state", halting_model.__dict__)
        self.assertNotIn(
            "finalize_weighted_accumulation",
            halting_model.__dict__,
        )
        self.assertIsNone(monitor._tracker_manager)
        self.assertEqual(monitor._halting_layers, [])
        self.assertEqual(monitor._survival_history, {})


if __name__ == "__main__":
    unittest.main()
