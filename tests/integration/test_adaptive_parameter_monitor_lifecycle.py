from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterMonitorCallback,
    BankExpansionFactorOptions,
    WeightBankUtilizationMonitorCallback,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def generator_config() -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=2,
        hidden_dim=2,
        output_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=2,
                output_dim=2,
                bias_flag=True,
            ),
        ),
    )


def adaptive_linear() -> torch.nn.Module:
    layer_config = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=2,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=WeightedBankDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
                model_config=generator_config(),
            )
        ),
    )
    layer = layer_config.build()
    bank = layer.adaptive_behaviour.bias_model
    generator = bank.model[0].model
    with torch.no_grad():
        layer.weight_params.copy_(torch.eye(2))
        layer.bias_params.copy_(torch.tensor([1.0, -1.0]))
        bank.weight_bank.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )
        generator.weight_params.zero_()
        generator.bias_params.zero_()
    return layer


class AdaptiveParameterTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        fail_after_forward: bool = False,
        learning_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.adaptive = adaptive_linear()
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.logged_calls: list[tuple[int, str, Tensor]] = []
        self.last_input: Tensor | None = None
        self.last_output: Tensor | None = None

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (inputs,) = batch
        output = self.adaptive(inputs)
        self.last_input = inputs.detach().clone()
        self.last_output = output.detach().clone()
        if self.fail_after_forward:
            raise RuntimeError("deliberate adaptive-parameter lifecycle failure")
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
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


def loader(num_batches: int = 1) -> DataLoader:
    values = torch.tensor(
        [
            [1.0, 2.0],
            [-3.0, 0.5],
            [0.25, -1.0],
            [2.0, 3.0],
            [-2.0, -0.5],
            [1.5, 0.75],
        ]
    )
    return DataLoader(
        TensorDataset(values[: num_batches * 2]),
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


class AdaptiveParameterMonitorLifecycleTests(unittest.TestCase):
    ADAPTIVE_PREFIX = "adaptive.adaptive_behaviour/bias/batch"
    BANK_PREFIX = "adaptive.adaptive_behaviour.bias_model/bank"

    def assert_logged_close(
        self,
        logged: dict[str, Tensor],
        name: str,
        expected: Tensor,
    ) -> None:
        self.assertIn(name, logged)
        torch.testing.assert_close(logged[name], expected)
        self.assertTrue(torch.isfinite(logged[name]), name)

    def test_real_trainer_logs_exact_metrics_visuals_and_updates_parameters(self):
        model = AdaptiveParameterTrainingModule()
        adaptive_callback = AdaptiveParameterMonitorCallback(
            log_every_n_steps=1,
            log_histograms=True,
        )
        bank_callback = WeightBankUtilizationMonitorCallback(
            log_every_n_steps=1,
            history_size=2,
            log_per_slot_scalars=True,
        )
        bank = model.adaptive.adaptive_behaviour.bias_model
        generator = bank.model[0].model
        initial_weight = model.adaptive.weight_params.detach().clone()
        initial_bank = bank.weight_bank.detach().clone()
        initial_generator_bias = generator.bias_params.detach().clone()

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="adaptive_parameters",
            )
            fit_trainer = trainer(
                Path(temporary_directory),
                [adaptive_callback, bank_callback],
                logger=tensorboard_logger,
            )
            fit_trainer.fit(model, train_dataloaders=loader())
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(tensorboard_logger.log_dir)
            events.Reload()
            tags = events.Tags()

        self.assertIsNotNone(model.last_input)
        self.assertIsNotNone(model.last_output)
        generated_bias = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
        expected_output = model.last_input + generated_bias
        torch.testing.assert_close(model.last_output, expected_output)

        logged = dict(fit_trainer.logged_metrics)
        base = torch.tensor([1.0, -1.0])
        delta = generated_bias - base
        expected_adaptive = {
            "output_mean": generated_bias.mean(),
            "output_var": generated_bias.var(unbiased=False),
            "output_min": generated_bias.min(),
            "output_max": generated_bias.max(),
            "output_l2_norm": generated_bias.norm(),
            "output_max_abs": generated_bias.abs().max(),
            "base_mean": base.mean(),
            "base_var": base.var(unbiased=False),
            "delta_mean": delta.mean(),
            "delta_var": delta.var(unbiased=False),
            "delta_l2_norm": delta.norm(),
            "relative_delta_norm": delta.norm() / base.norm(),
            "cross_sample_std": torch.tensor(0.0),
            "adaptivity_ratio": torch.tensor(0.0),
            "centroid_cosine_mean": torch.tensor(1.0),
            "weight_bank_mean": torch.tensor(2.5),
            "weight_bank_var": torch.tensor(1.25),
            "weight_bank_l2_norm": torch.tensor(math.sqrt(30.0)),
        }
        for suffix, expected in expected_adaptive.items():
            self.assert_logged_close(
                logged,
                f"{self.ADAPTIVE_PREFIX}/{suffix}",
                expected,
            )

        expected_bank = {
            "selection_entropy_marginal": torch.tensor(math.log(2.0)),
            "selection_entropy_mean": torch.tensor(math.log(2.0)),
            "utilization_coefficient_of_variation": torch.tensor(0.0),
            "active_slots": torch.tensor(2.0),
            "dead_slot_fraction": torch.tensor(0.0),
            "max_utilization": torch.tensor(0.5),
            "min_utilization": torch.tensor(0.5),
            "slot_0/utilization": torch.tensor(0.5),
            "slot_1/utilization": torch.tensor(0.5),
        }
        for suffix, expected in expected_bank.items():
            self.assert_logged_close(
                logged,
                f"{self.BANK_PREFIX}/{suffix}",
                expected,
            )

        self.assertIn(
            f"{self.ADAPTIVE_PREFIX}/output",
            tags["histograms"],
        )
        self.assertIn(
            f"{self.ADAPTIVE_PREFIX}/delta",
            tags["histograms"],
        )
        self.assertIn(
            f"{self.BANK_PREFIX}/histogram/utilization",
            tags["histograms"],
        )
        self.assertIn(
            f"{self.BANK_PREFIX}/heatmap/utilization",
            tags["images"],
        )
        self.assertFalse(
            torch.equal(model.adaptive.weight_params.detach(), initial_weight)
        )
        self.assertFalse(torch.equal(bank.weight_bank.detach(), initial_bank))
        self.assertFalse(
            torch.equal(generator.bias_params.detach(), initial_generator_bias)
        )
        self.assertEqual(adaptive_callback._hooks, [])
        self.assertEqual(bank_callback._hooks, [])
        self.assertEqual(bank_callback._bank_modules, [])
        self.assertEqual(bank_callback._utilization_history, {})
        self.assertEqual(bank_callback._last_bank_logits, {})
        self.assertEqual(bank._forward_hooks, {})
        self.assertEqual(bank.model._forward_hooks, {})

    def test_real_trainer_applies_monitor_cadence_without_duplicate_emission(self):
        model = AdaptiveParameterTrainingModule(learning_rate=0.0)
        adaptive_callback = AdaptiveParameterMonitorCallback(log_every_n_steps=2)
        bank_callback = WeightBankUtilizationMonitorCallback(log_every_n_steps=2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(
                Path(temporary_directory),
                [adaptive_callback, bank_callback],
            )
            fit_trainer.fit(model, train_dataloaders=loader(num_batches=3))

        adaptive_name = f"{self.ADAPTIVE_PREFIX}/output_mean"
        adaptive_calls = [
            (step, name)
            for step, name, _ in model.logged_calls
            if name == adaptive_name
        ]
        self.assertEqual(adaptive_calls, [(0, adaptive_name), (2, adaptive_name)])
        bank_name = f"{self.BANK_PREFIX}/selection_entropy_marginal"
        bank_calls = [
            (step, name) for step, name, _ in model.logged_calls if name == bank_name
        ]
        self.assertEqual(len(bank_calls), 2)
        self.assertEqual([name for _, name in bank_calls], [bank_name, bank_name])
        self.assertEqual(adaptive_callback._hooks, [])
        self.assertEqual(bank_callback._hooks, [])

    def test_real_trainer_exception_cleans_both_monitor_callbacks(self):
        model = AdaptiveParameterTrainingModule(
            fail_after_forward=True,
            learning_rate=0.0,
        )
        adaptive_callback = AdaptiveParameterMonitorCallback(log_every_n_steps=1)
        bank_callback = WeightBankUtilizationMonitorCallback(log_every_n_steps=1)
        augmentation = model.adaptive.adaptive_behaviour
        bank = augmentation.bias_model

        with tempfile.TemporaryDirectory() as temporary_directory:
            fit_trainer = trainer(
                Path(temporary_directory),
                [adaptive_callback, bank_callback],
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "^deliberate adaptive-parameter lifecycle failure$",
            ):
                fit_trainer.fit(model, train_dataloaders=loader())

        self.assertEqual(adaptive_callback._hooks, [])
        self.assertEqual(bank_callback._hooks, [])
        self.assertEqual(bank_callback._bank_modules, [])
        self.assertEqual(bank_callback._utilization_history, {})
        self.assertEqual(bank_callback._last_bank_logits, {})
        self.assertEqual(augmentation._forward_hooks, {})
        self.assertEqual(bank._forward_hooks, {})
        self.assertEqual(bank.model._forward_hooks, {})


if __name__ == "__main__":
    unittest.main()
