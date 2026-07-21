from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerControllerMonitorCallback,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    RecurrentLayerMonitorCallback,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.nn import Module
from support.monitor import same_bound_method


def linear_stack_config(dim: int, *, bias_flag: bool = True) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=dim,
                output_dim=dim,
                bias_flag=bias_flag,
            ),
        ),
    )


def set_linear_identity(layer: Layer) -> None:
    with torch.no_grad():
        layer.model.weight_params.copy_(
            torch.eye(
                layer.input_dim,
                layer.output_dim,
                device=layer.model.weight_params.device,
                dtype=layer.model.weight_params.dtype,
            )
        )
        if layer.model.bias_params is not None:
            layer.model.bias_params.zero_()


def controlled_layer() -> Layer:
    layer = Layer(
        LayerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.TANH,
            residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
            dropout_probability=1.0,
            layer_norm_position=LayerNormPositionOptions.BEFORE,
            gate_config=GateConfig(
                gate_dim=2,
                option=LayerGateOptions.MULTIPLIER,
                activation=ActivationOptions.SIGMOID,
                model_config=linear_stack_config(2),
            ),
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=2,
                output_dim=2,
                bias_flag=True,
            ),
        )
    )
    set_linear_identity(layer)
    gate_layer = layer.gate_model.model[0]
    with torch.no_grad():
        gate_layer.model.weight_params.zero_()
        gate_layer.model.bias_params.zero_()
    return layer


def trainable_layer() -> Layer:
    layer = Layer(
        LayerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=2,
                output_dim=2,
                bias_flag=False,
            ),
        )
    )
    set_linear_identity(layer)
    return layer


@dataclass
class TrainableIncrementConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    increment: float | None = optional_field("Increment per recurrent step.")

    def _registry_owner(self) -> type:
        return TrainableIncrementBlock


class TrainableIncrementBlock(Module):
    def __init__(
        self,
        cfg: TrainableIncrementConfig,
        overrides: TrainableIncrementConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, state: LayerState) -> LayerState:
        state.hidden = state.hidden + self.scale * self.cfg.increment
        return state


def recurrent_layer() -> RecurrentLayer:
    recurrent = RecurrentLayer(
        RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=3,
            recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
            block_config=TrainableIncrementConfig(
                input_dim=2,
                output_dim=2,
                increment=0.25,
            ),
            gate_config=GateConfig(
                gate_dim=2,
                option=LayerGateOptions.MULTIPLIER,
                activation=ActivationOptions.SIGMOID,
                model_config=linear_stack_config(2),
            ),
            residual_config=None,
            halting_config=None,
            memory_config=None,
        )
    )
    gate_layer = recurrent.recurrent_gate.model[0]
    with torch.no_grad():
        gate_layer.model.weight_params.zero_()
        gate_layer.model.bias_params.zero_()
    return recurrent


class LayerTrainingModule(LightningModule):
    def __init__(self, *, fail_after_forward: bool = False) -> None:
        super().__init__()
        self.controlled = controlled_layer()
        self.trainable = trainable_layer()
        self.fail_after_forward = fail_after_forward
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_input: Tensor | None = None
        self.last_controlled_output: Tensor | None = None

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (hidden,) = batch
        controlled_output = self.controlled(LayerState(hidden=hidden)).hidden
        trainable_output = self.trainable(LayerState(hidden=hidden)).hidden
        self.last_input = hidden.detach().clone()
        self.last_controlled_output = controlled_output.detach().clone()
        if self.fail_after_forward:
            raise RuntimeError("deliberate layer lifecycle failure")
        return controlled_output.square().mean() + trainable_output.square().mean()

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if torch.is_tensor(value):
            self.logged_calls.append(
                (int(self.global_step), name, value.detach().clone())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.05)


class RecurrentTrainingModule(LightningModule):
    def __init__(self, *, fail_after_forward: bool = False) -> None:
        super().__init__()
        self.recurrent = recurrent_layer()
        self.fail_after_forward = fail_after_forward
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_output: Tensor | None = None

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (hidden,) = batch
        output = self.recurrent(LayerState(hidden=hidden)).hidden
        self.last_output = output.detach().clone()
        if self.fail_after_forward:
            raise RuntimeError("deliberate recurrent lifecycle failure")
        return output.square().mean()

    def log(
        self,
        name: str,
        value: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if torch.is_tensor(value):
            self.logged_calls.append(
                (int(self.global_step), name, value.detach().clone())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.05)


def layer_loader() -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(
                [
                    [1.0, 3.0],
                    [-2.0, 2.0],
                ]
            )
        ),
        batch_size=2,
        shuffle=False,
    )


def recurrent_loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.zeros(2, 2)),
        batch_size=2,
        shuffle=False,
    )


def trainer(
    root: Path,
    callbacks: list,
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


class LayerMonitorLifecycleTests(unittest.TestCase):
    def test_real_trainer_logs_exact_layer_controller_metrics_and_updates(
        self,
    ) -> None:
        model = LayerTrainingModule()
        callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        initial_weight = model.trainable.model.weight_params.detach().clone()
        original_activation = model.controlled._Layer__maybe_apply_activation
        original_residual = model.controlled._Layer__maybe_apply_residual_connection

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [callback])
            fit_trainer.fit(model, train_dataloaders=layer_loader())

        self.assertIsNotNone(model.last_input)
        self.assertIsNotNone(model.last_controlled_output)
        hidden = model.last_input
        torch.testing.assert_close(model.last_controlled_output, hidden)
        normalized = torch.layer_norm(hidden, (2,))
        activated = torch.tanh(normalized)
        zeros = torch.zeros_like(activated)
        expected = {
            "controlled/gate/output_mean": torch.tensor(0.0),
            "controlled/gate/output_var": torch.tensor(0.0),
            "controlled/gate/positive_fraction": torch.tensor(0.0),
            "controlled/gate/saturation_fraction": torch.tensor(0.0),
            "controlled/gate/effective_mean": torch.tensor(0.5),
            "controlled/gate/effective_var": torch.tensor(0.0),
            "controlled/gate/effective_positive_fraction": torch.tensor(1.0),
            "controlled/gate/effective_saturation_fraction": torch.tensor(0.0),
            "controlled/dropout/zero_fraction": torch.tensor(1.0),
            "controlled/dropout/dropped_nonzero_fraction": torch.tensor(1.0),
            "controlled/layer_norm/output_mean": normalized.mean(),
            "controlled/layer_norm/output_var": normalized.var(unbiased=False),
            "controlled/layer_norm/relative_delta_norm": (
                (normalized - hidden).norm() / hidden.norm().clamp_min(1e-6)
            ),
            "controlled/activation/zero_fraction": torch.tensor(0.0),
            "controlled/activation/saturation_fraction": (
                ((activated < -0.99) | (activated > 0.99)).float().mean()
            ),
            "controlled/residual/contribution_ratio": torch.tensor(1.0),
            "controlled/residual/input_ratio": (
                hidden.norm() / zeros.norm().clamp_min(1e-6)
            ),
        }
        logged = dict(fit_trainer.logged_metrics)
        for name, expected_value in expected.items():
            self.assertIn(name, logged)
            torch.testing.assert_close(logged[name], expected_value)
            self.assertTrue(torch.isfinite(logged[name]), name)

        self.assertFalse(
            torch.equal(model.trainable.model.weight_params.detach(), initial_weight)
        )
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._hooked_gate_model_ids, set())
        self.assertTrue(
            same_bound_method(
                model.controlled._Layer__maybe_apply_activation,
                original_activation,
            )
        )
        self.assertTrue(
            same_bound_method(
                model.controlled._Layer__maybe_apply_residual_connection,
                original_residual,
            )
        )

    def test_real_trainer_logs_exact_recurrent_metrics_visuals_and_updates(
        self,
    ) -> None:
        model = RecurrentTrainingModule()
        callback = RecurrentLayerMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
            log_per_step_scalars=True,
        )
        initial_scale = model.recurrent.block_model.scale.detach().clone()
        original_forward = model.recurrent.forward
        original_controllers = model.recurrent._RecurrentLayer__run_controllers

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="layers",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [callback],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=recurrent_loader())
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()
            tags = events.Tags()

        delta_means = torch.tensor(
            [
                math.sqrt(2.0) * 0.125,
                math.sqrt(2.0) * 0.0625,
                math.sqrt(2.0) * 0.03125,
            ]
        )
        expected = {
            "recurrent/recurrent/actual_steps": torch.tensor(3.0),
            "recurrent/recurrent/hidden_delta_mean": delta_means.mean(),
            "recurrent/recurrent/hidden_delta_max": delta_means[0],
            "recurrent/recurrent/hidden_delta_final": delta_means[-1],
            "recurrent/recurrent/convergence_ratio": torch.tensor(0.25),
            "recurrent/recurrent/max_step_fraction": torch.tensor(1.0),
            "recurrent/recurrent/gate/open_mean": torch.tensor(0.5),
            "recurrent/recurrent/gate/open_fraction": torch.tensor(0.0),
            "recurrent/recurrent/gate/saturation_fraction": torch.tensor(0.0),
        }
        expected.update(
            {
                f"recurrent/recurrent/step_{index}/hidden_delta_mean": value
                for index, value in enumerate(delta_means)
            }
        )
        logged = dict(fit_trainer.logged_metrics)
        for name, expected_value in expected.items():
            self.assertIn(name, logged)
            torch.testing.assert_close(logged[name], expected_value)
            self.assertTrue(torch.isfinite(logged[name]), name)

        torch.testing.assert_close(
            model.last_output,
            torch.full((2, 2), 0.21875),
        )
        self.assertFalse(
            torch.equal(model.recurrent.block_model.scale.detach(), initial_scale)
        )
        self.assertIn(
            "recurrent/recurrent/histogram/hidden_delta",
            tags["histograms"],
        )
        self.assertIn(
            "recurrent/recurrent/heatmap/hidden_delta_by_step",
            tags["images"],
        )
        self.assertEqual(callback._hooks, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._observations, {})
        self.assertEqual(callback._delta_history, {})
        self.assertEqual(callback._latest_gate_logits, {})
        self.assertTrue(same_bound_method(model.recurrent.forward, original_forward))
        self.assertTrue(
            same_bound_method(
                model.recurrent._RecurrentLayer__run_controllers,
                original_controllers,
            )
        )

    def test_real_trainer_exception_restores_both_monitor_types(self) -> None:
        layer_model = LayerTrainingModule(fail_after_forward=True)
        layer_callback = LayerControllerMonitorCallback(log_every_n_steps=1)
        layer_activation = layer_model.controlled._Layer__maybe_apply_activation
        layer_residual = layer_model.controlled._Layer__maybe_apply_residual_connection

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [layer_callback])
            with self.assertRaisesRegex(
                RuntimeError,
                r"^deliberate layer lifecycle failure$",
            ):
                fit_trainer.fit(layer_model, train_dataloaders=layer_loader())

        self.assertEqual(layer_callback._hooks, [])
        self.assertEqual(layer_callback._wrapped_methods, [])
        self.assertTrue(
            same_bound_method(
                layer_model.controlled._Layer__maybe_apply_activation,
                layer_activation,
            )
        )
        self.assertTrue(
            same_bound_method(
                layer_model.controlled._Layer__maybe_apply_residual_connection,
                layer_residual,
            )
        )

        recurrent_model = RecurrentTrainingModule(fail_after_forward=True)
        recurrent_callback = RecurrentLayerMonitorCallback(log_every_n_steps=1)
        recurrent_forward = recurrent_model.recurrent.forward
        recurrent_controllers = (
            recurrent_model.recurrent._RecurrentLayer__run_controllers
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(Path(temporary_directory), [recurrent_callback])
            with self.assertRaisesRegex(
                RuntimeError,
                r"^deliberate recurrent lifecycle failure$",
            ):
                fit_trainer.fit(
                    recurrent_model,
                    train_dataloaders=recurrent_loader(),
                )

        self.assertEqual(recurrent_callback._hooks, [])
        self.assertEqual(recurrent_callback._wrapped_methods, [])
        self.assertEqual(recurrent_callback._observations, {})
        self.assertEqual(recurrent_callback._delta_history, {})
        self.assertEqual(recurrent_callback._latest_gate_logits, {})
        self.assertTrue(
            same_bound_method(
                recurrent_model.recurrent.forward,
                recurrent_forward,
            )
        )
        self.assertTrue(
            same_bound_method(
                recurrent_model.recurrent._RecurrentLayer__run_controllers,
                recurrent_controllers,
            )
        )


if __name__ == "__main__":
    unittest.main()
