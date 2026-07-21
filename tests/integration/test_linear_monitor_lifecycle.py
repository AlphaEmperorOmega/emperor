from __future__ import annotations

import math
import unittest

import torch
from lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from emperor.linears import LinearLayer, LinearLayerConfig, LinearMonitorCallback


class _LinearRegressionExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=True)
        )

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        return F.mse_loss(self.linear(inputs), targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _TupleOutputLinear(LinearLayer):
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        return (super().forward(X),)


class _TupleOutputExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = _TupleOutputLinear(
            LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=True)
        )

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        (prediction,) = self.linear(inputs)
        return F.mse_loss(prediction, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _NoInputLinear(LinearLayer):
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:
        return self.weight_params.sum().reshape(1)


class _NoInputExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = _NoInputLinear(
            LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=False)
        )

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.linear().sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _FailingLinearExperiment(_LinearRegressionExperiment):
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, _ = batch
        self.linear(inputs)
        raise RuntimeError("deliberate training failure")


class _ExactDiagnosticsExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=2, bias_flag=True)
        )
        with torch.no_grad():
            self.linear.weight_params.copy_(
                torch.tensor(
                    [
                        [0.4, 0.0],
                        [0.0, 0.1],
                    ]
                )
            )
            self.linear.bias_params.copy_(torch.tensor([0.25, -0.5]))

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        return self.linear(inputs).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _DeltaDiagnosticsExperiment(_ExactDiagnosticsExperiment):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.linear.weight_params.copy_(
                torch.tensor(
                    [
                        [0.2, 0.0],
                        [0.0, 0.05],
                    ]
                )
            )
            self.linear.bias_params.copy_(torch.tensor([0.3, -0.2]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _ChangingBiasExperiment(_LinearRegressionExperiment):
    def on_train_batch_start(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx == 1:
            self.linear.bias_params = None
        elif batch_idx == 2:
            self.linear.bias_params = torch.nn.Parameter(torch.tensor([0.75]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _TwoLinearExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=2, bias_flag=False)
        )
        self.head = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=1, bias_flag=False)
        )
        with torch.no_grad():
            self.encoder.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                    ]
                )
            )
            self.head.weight_params.copy_(torch.tensor([[3.0], [-1.0]]))

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        return self.head(self.encoder(inputs)).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _CapturingLogExperiment(_LinearRegressionExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.logged_calls: list[tuple[int, str]] = []

    def log(self, name: str, value: object, *args: object, **kwargs: object) -> None:
        self.logged_calls.append((self.global_step, name))
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.0)


class _AccumulationExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = LinearLayer(
            LinearLayerConfig(input_dim=1, output_dim=1, bias_flag=False)
        )
        with torch.no_grad():
            self.linear.weight_params.fill_(1.0)
        self.logged_scalar_calls: list[tuple[int, str, float]] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        return self.linear(inputs).sum()

    def log(self, name: str, value: object, *args: object, **kwargs: object) -> None:
        if torch.is_tensor(value) and value.numel() == 1:
            self.logged_scalar_calls.append(
                (self.global_step, name, value.detach().cpu().item())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _KeywordInputExperiment(_LinearRegressionExperiment):
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        return F.mse_loss(self.linear(X=inputs), targets)


class _ReplacingLinearExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = LinearLayer(
            LinearLayerConfig(input_dim=1, output_dim=1, bias_flag=False)
        )
        with torch.no_grad():
            self.linear.weight_params.fill_(1.0)
        self.removed_layer: LinearLayer | None = None
        self.logged_scalar_calls: list[tuple[int, str, float]] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        return self.linear(inputs).sum()

    def on_train_batch_start(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> None:
        if batch_idx != 1:
            return
        self.removed_layer = self.linear
        replacement = LinearLayer(
            LinearLayerConfig(input_dim=1, output_dim=1, bias_flag=False)
        )
        with torch.no_grad():
            replacement.weight_params.fill_(2.0)
        self.linear = replacement
        optimizer = self.optimizers(use_pl_optimizer=False)
        optimizer.param_groups[0]["params"] = list(replacement.parameters())

    def log(self, name: str, value: object, *args: object, **kwargs: object) -> None:
        if torch.is_tensor(value) and value.numel() == 1:
            self.logged_scalar_calls.append(
                (self.global_step, name, value.detach().cpu().item())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _ManualTwoOptimizerExperiment(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False
        config = LinearLayerConfig(input_dim=1, output_dim=1, bias_flag=False)
        self.first = LinearLayer(config)
        self.second = LinearLayer(config)
        with torch.no_grad():
            self.first.weight_params.fill_(1.0)
            self.second.weight_params.fill_(1.0)
        self.logged_scalar_calls: list[tuple[int, str, float]] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        (inputs,) = batch
        first_optimizer, second_optimizer = self.optimizers()

        first_optimizer.zero_grad()
        first_loss = self.first(inputs).sum()
        self.manual_backward(first_loss)
        first_optimizer.step()

        second_optimizer.zero_grad()
        second_loss = self.second(inputs).sum()
        self.manual_backward(second_loss)
        second_optimizer.step()
        return first_loss.detach() + second_loss.detach()

    def log(self, name: str, value: object, *args: object, **kwargs: object) -> None:
        if torch.is_tensor(value) and value.numel() == 1:
            self.logged_scalar_calls.append(
                (self.global_step, name, value.detach().cpu().item())
            )
        super().log(name, value, *args, **kwargs)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        return [
            torch.optim.SGD(self.first.parameters(), lr=0.2),
            torch.optim.SGD(self.second.parameters(), lr=0.2),
        ]


class _TrainAndValidationActivationExperiment(_CapturingLogExperiment):
    def on_validation_model_eval(self) -> None:
        pass

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        return F.mse_loss(self.linear(inputs), targets)


class LinearMonitorLifecycleTests(unittest.TestCase):
    def test_real_trainer_logs_exact_forward_parameter_gradient_and_health_metrics(
        self,
    ) -> None:
        model = _ExactDiagnosticsExperiment()
        inputs = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        monitor = LinearMonitorCallback(
            log_every_n_steps=1,
            log_weight_conditioning=True,
        )
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(TensorDataset(inputs), batch_size=2),
        )

        effective_rank = math.exp(-(0.8 * math.log(0.8) + 0.2 * math.log(0.2)))
        expected_metrics = {
            "linear/input/mean": 2.5,
            "linear/input/var": 1.25,
            "linear/output/mean": 0.425,
            "linear/output/var": 0.475625,
            "linear/weights/mean": 0.125,
            "linear/weights/var": 0.026875,
            "linear/weights/l2_norm": math.sqrt(0.17),
            "linear/bias/mean": -0.125,
            "linear/bias/var": 0.140625,
            "linear/bias/l2_norm": math.sqrt(0.3125),
            "linear/weights/grad_mean": 5.0,
            "linear/weights/grad_var": 1.0,
            "linear/weights/grad_norm": math.sqrt(104.0),
            "linear/weights/gradient_to_weight_norm_ratio": (
                math.sqrt(104.0) / math.sqrt(0.17)
            ),
            "linear/weights/delta_norm": 0.0,
            "linear/weights/relative_delta_norm": 0.0,
            "linear/weights/update_ratio": 0.0,
            "linear/bias/grad_mean": 2.0,
            "linear/bias/grad_var": 0.0,
            "linear/bias/grad_norm": math.sqrt(8.0),
            "linear/bias/delta_norm": 0.0,
            "linear/bias/relative_delta_norm": 0.0,
            "linear/weights/dead_input_fraction": 0.0,
            "linear/weights/dead_output_fraction": 0.0,
            "linear/weights/spectral_norm": 0.4,
            "linear/weights/condition_number": 4.0,
            "linear/weights/effective_rank": effective_rank,
        }
        self.assertEqual(set(trainer.logged_metrics), set(expected_metrics))
        for metric_name, expected_value in expected_metrics.items():
            self.assertAlmostEqual(
                trainer.logged_metrics[metric_name].item(),
                expected_value,
                places=5,
                msg=metric_name,
            )

    def test_real_optimizer_steps_produce_exact_parameter_delta_metrics(self) -> None:
        model = _DeltaDiagnosticsExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.eye(2)),
                batch_size=1,
                shuffle=False,
            ),
        )

        torch.testing.assert_close(
            model.linear.weight_params,
            torch.tensor(
                [
                    [0.1, -0.1],
                    [-0.1, -0.05],
                ]
            ),
        )
        torch.testing.assert_close(
            model.linear.bias_params,
            torch.tensor([0.1, -0.4]),
        )
        expected_delta_metrics = {
            "linear/weights/delta_norm": math.sqrt(0.02),
            "linear/weights/relative_delta_norm": math.sqrt(0.02) / 0.15,
            "linear/weights/update_ratio": math.sqrt(0.02) / 0.15,
            "linear/weights/gradient_to_weight_norm_ratio": (math.sqrt(2.0) / 0.15),
            "linear/bias/delta_norm": math.sqrt(0.02),
            "linear/bias/relative_delta_norm": math.sqrt(0.02) / math.sqrt(0.13),
        }
        for metric_name, expected_value in expected_delta_metrics.items():
            self.assertAlmostEqual(
                trainer.logged_metrics[metric_name].item(),
                expected_value,
                places=5,
                msg=metric_name,
            )

    def test_dead_feature_metrics_include_norms_exactly_at_relative_threshold(
        self,
    ) -> None:
        model = _ExactDiagnosticsExperiment()
        with torch.no_grad():
            model.linear.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [1999.0, 0.0],
                    ]
                )
            )
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.tensor([[1.0, 1.0]])),
                batch_size=1,
            ),
        )

        self.assertEqual(
            trainer.logged_metrics["linear/weights/dead_input_fraction"].item(),
            0.5,
        )
        self.assertEqual(
            trainer.logged_metrics["linear/weights/dead_output_fraction"].item(),
            0.5,
        )

    def test_replacing_a_bias_does_not_compare_it_with_the_removed_parameter(
        self,
    ) -> None:
        model = _ChangingBiasExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        inputs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        targets = torch.zeros(3, 1)

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(inputs, targets),
                batch_size=1,
                shuffle=False,
            ),
        )

        self.assertEqual(
            trainer.logged_metrics["linear/bias/delta_norm"].item(),
            0.0,
        )
        self.assertEqual(
            trainer.logged_metrics["linear/bias/relative_delta_norm"].item(),
            0.0,
        )
        torch.testing.assert_close(
            model.linear.bias_params,
            torch.tensor([0.75]),
        )

    def test_real_trainer_tracks_every_linear_under_its_module_namespace(
        self,
    ) -> None:
        model = _TwoLinearExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.tensor([[1.0, 2.0]])),
                batch_size=1,
            ),
        )

        expected_metrics = {
            "encoder/input/mean": 1.5,
            "encoder/output/mean": 2.5,
            "encoder/weights/mean": 0.75,
            "head/input/mean": 2.5,
            "head/output/mean": -1.0,
            "head/weights/mean": 1.0,
        }
        for metric_name, expected_value in expected_metrics.items():
            self.assertAlmostEqual(
                trainer.logged_metrics[metric_name].item(),
                expected_value,
                places=6,
                msg=metric_name,
            )
        self.assertEqual(len(model.encoder._forward_hooks), 0)
        self.assertEqual(len(model.head._forward_hooks), 0)

    def test_real_trainer_cadence_uses_global_step_for_each_metric_family(
        self,
    ) -> None:
        model = _CapturingLogExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=2)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(
                    torch.tensor(
                        [
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0],
                        ]
                    ),
                    torch.zeros(3, 1),
                ),
                batch_size=1,
                shuffle=False,
            ),
        )

        input_mean_steps = [
            step for step, name in model.logged_calls if name == "linear/input/mean"
        ]
        weight_mean_steps = [
            step for step, name in model.logged_calls if name == "linear/weights/mean"
        ]
        self.assertEqual(input_mean_steps, [2])
        self.assertEqual(weight_mean_steps, [2])

    def test_gradient_accumulation_emits_one_exact_sample_per_optimizer_step(
        self,
    ) -> None:
        model = _AccumulationExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            accumulate_grad_batches=2,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.tensor([[1.0], [3.0], [5.0], [7.0]])),
                batch_size=1,
                shuffle=False,
            ),
        )

        expected_calls = {
            "linear/input/mean": ((1, 2.0), (2, 6.0)),
            "linear/input/var": ((1, 1.0), (2, 1.0)),
            "linear/output/mean": ((1, 2.0), (2, 4.8)),
            "linear/output/var": ((1, 1.0), (2, 0.64)),
            "linear/weights/grad_mean": ((1, 2.0), (2, 6.0)),
            "linear/weights/grad_norm": ((1, 2.0), (2, 6.0)),
            "linear/weights/mean": ((1, 0.8), (2, 0.2)),
            "linear/weights/delta_norm": ((1, 0.2), (2, 0.6)),
            "linear/weights/relative_delta_norm": ((1, 0.2), (2, 0.75)),
            "linear/weights/update_ratio": ((1, 0.2), (2, 0.75)),
        }
        for metric_name, expected in expected_calls.items():
            with self.subTest(metric_name=metric_name):
                actual = [
                    (step, value)
                    for step, name, value in model.logged_scalar_calls
                    if name == metric_name
                ]
                self.assertEqual(len(actual), 2)
                for (actual_step, actual_value), (
                    expected_step,
                    expected_value,
                ) in zip(actual, expected, strict=True):
                    self.assertEqual(actual_step, expected_step)
                    self.assertAlmostEqual(actual_value, expected_value, places=5)
        self.assertAlmostEqual(model.linear.weight_params.item(), 0.2, places=5)

    def test_manual_optimizers_emit_each_linear_update_without_overwriting(
        self,
    ) -> None:
        model = _ManualTwoOptimizerExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.ones(1, 1)),
                batch_size=1,
            ),
        )

        expected_calls = {
            "first/input/mean": [(1, 1.0)],
            "first/weights/delta_norm": [(1, 0.2), (2, 0.0)],
            "second/input/mean": [(2, 1.0)],
            "second/weights/delta_norm": [(1, 0.0), (2, 0.2)],
        }
        for metric_name, expected in expected_calls.items():
            with self.subTest(metric_name=metric_name):
                actual = [
                    (step, value)
                    for step, name, value in model.logged_scalar_calls
                    if name == metric_name
                ]
                self.assertEqual(len(actual), len(expected))
                for (actual_step, actual_value), (
                    expected_step,
                    expected_value,
                ) in zip(actual, expected, strict=True):
                    self.assertEqual(actual_step, expected_step)
                    self.assertAlmostEqual(actual_value, expected_value, places=6)
        self.assertAlmostEqual(model.first.weight_params.item(), 0.8, places=6)
        self.assertAlmostEqual(model.second.weight_params.item(), 0.8, places=6)

    def test_real_trainer_tracks_keyword_input_activations(self) -> None:
        model = _KeywordInputExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    torch.zeros(2, 1),
                ),
                batch_size=2,
            ),
        )

        self.assertEqual(trainer.logged_metrics["linear/input/mean"].item(), 2.5)
        self.assertAlmostEqual(
            trainer.logged_metrics["linear/input/var"].item(),
            1.25,
            places=6,
        )

    def test_real_trainer_refreshes_a_replaced_linear_between_batches(self) -> None:
        model = _ReplacingLinearExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.ones(2, 1)),
                batch_size=1,
                shuffle=False,
            ),
        )

        output_means = [
            (step, value)
            for step, name, value in model.logged_scalar_calls
            if name == "linear/output/mean"
        ]
        self.assertEqual(output_means, [(1, 1.0), (2, 2.0)])
        for metric_name, expected_values in (
            ("linear/weights/mean", (0.9, 1.9)),
            ("linear/weights/grad_norm", (1.0, 1.0)),
            ("linear/weights/delta_norm", (0.1, 0.1)),
        ):
            actual_values = [
                value
                for _, name, value in model.logged_scalar_calls
                if name == metric_name
            ]
            self.assertEqual(len(actual_values), 2)
            for actual, expected in zip(actual_values, expected_values, strict=True):
                self.assertAlmostEqual(actual, expected, places=6)
        self.assertIsNotNone(model.removed_layer)
        self.assertEqual(len(model.removed_layer._forward_hooks), 0)
        self.assertEqual(len(model.linear._forward_hooks), 0)
        self.assertEqual(monitor._hooks, {})
        self.assertEqual(monitor._linear_modules, {})

    def test_real_trainer_logs_activation_metrics_only_for_training_batches(
        self,
    ) -> None:
        model = _TrainAndValidationActivationExperiment()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=1,
        )
        training_data = DataLoader(
            TensorDataset(
                torch.ones(1, 2),
                torch.zeros(1, 1),
            ),
            batch_size=1,
        )
        validation_data = DataLoader(
            TensorDataset(
                torch.full((1, 2), 10.0),
                torch.zeros(1, 1),
            ),
            batch_size=1,
        )

        trainer.fit(
            model,
            train_dataloaders=training_data,
            val_dataloaders=validation_data,
        )

        for metric_name in (
            "linear/input/mean",
            "linear/input/var",
            "linear/output/mean",
            "linear/output/var",
        ):
            with self.subTest(metric_name=metric_name):
                metric_steps = [
                    step for step, name in model.logged_calls if name == metric_name
                ]
                self.assertEqual(metric_steps, [1])
        self.assertEqual(
            trainer.logged_metrics["linear/input/mean"].item(),
            1.0,
        )

    def test_real_trainer_fit_logs_diagnostics_updates_parameters_and_cleans_up(
        self,
    ) -> None:
        torch.manual_seed(41)
        model = _LinearRegressionExperiment()
        inputs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
            ]
        )
        targets = torch.tensor([[2.0], [-1.0], [1.0], [5.0]])
        dataloader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=2,
            shuffle=False,
        )
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        initial_weight = model.linear.weight_params.detach().clone()
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            deterministic=True,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, train_dataloaders=dataloader)

        self.assertFalse(
            torch.equal(initial_weight, model.linear.weight_params.detach())
        )
        expected_metrics = {
            "linear/input/mean",
            "linear/input/var",
            "linear/output/mean",
            "linear/output/var",
            "linear/weights/mean",
            "linear/weights/var",
            "linear/weights/l2_norm",
            "linear/weights/delta_norm",
            "linear/weights/relative_delta_norm",
            "linear/weights/grad_norm",
            "linear/weights/gradient_to_weight_norm_ratio",
            "linear/weights/update_ratio",
        }
        self.assertTrue(expected_metrics.issubset(trainer.logged_metrics))
        self.assertNotIn("linear/weights/spectral_norm", trainer.logged_metrics)
        self.assertNotIn("linear/weights/condition_number", trainer.logged_metrics)
        self.assertNotIn("linear/weights/effective_rank", trainer.logged_metrics)
        for metric_name in expected_metrics:
            self.assertTrue(
                torch.isfinite(trainer.logged_metrics[metric_name]).all(),
                metric_name,
            )
        self.assertEqual(monitor._hooks, {})
        self.assertEqual(monitor._linear_modules, {})
        self.assertEqual(monitor._activation_moments, {})
        self.assertIsNone(monitor._pending_step)

    def test_real_trainer_logs_input_metrics_for_non_tensor_hook_output(
        self,
    ) -> None:
        model = _TupleOutputExperiment()
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0], [2.0]]),
            ),
            batch_size=2,
        )
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, train_dataloaders=dataloader)

        self.assertEqual(trainer.logged_metrics["linear/input/mean"].item(), 2.5)
        self.assertAlmostEqual(
            trainer.logged_metrics["linear/input/var"].item(),
            1.25,
            places=6,
        )
        self.assertNotIn("linear/output/mean", trainer.logged_metrics)
        self.assertIn("linear/weights/l2_norm", trainer.logged_metrics)
        self.assertEqual(monitor._hooks, {})

    def test_real_trainer_logs_output_metrics_without_tensor_hook_input(
        self,
    ) -> None:
        model = _NoInputExperiment()
        expected_output = model.linear.weight_params.detach().sum().item()
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(torch.tensor([[0.0]])),
                batch_size=1,
            ),
        )

        self.assertNotIn("linear/input/mean", trainer.logged_metrics)
        self.assertAlmostEqual(
            trainer.logged_metrics["linear/output/mean"].item(),
            expected_output,
            places=6,
        )
        self.assertEqual(trainer.logged_metrics["linear/output/var"].item(), 0.0)
        self.assertIn("linear/weights/l2_norm", trainer.logged_metrics)
        self.assertEqual(monitor._hooks, {})

    def test_real_trainer_exception_cleans_up_callback_state(self) -> None:
        model = _FailingLinearExperiment()
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0], [2.0]]),
            ),
            batch_size=2,
        )
        monitor = LinearMonitorCallback(log_every_n_steps=1)
        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            callbacks=[monitor],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

        with self.assertRaisesRegex(RuntimeError, "deliberate training failure"):
            trainer.fit(model, train_dataloaders=dataloader)

        self.assertEqual(monitor._hooks, {})
        self.assertEqual(monitor._linear_modules, {})
        self.assertEqual(monitor._activation_moments, {})
        self.assertIsNone(monitor._pending_step)
        self.assertEqual(len(model.linear._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
