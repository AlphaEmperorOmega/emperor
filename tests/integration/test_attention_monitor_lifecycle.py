from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from emperor.attention import (
    AttentionMonitorCallback,
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from support.attention import build_attention_config


def _same_bound_method(left: object, right: object) -> bool:
    return getattr(left, "__self__", None) is getattr(
        right,
        "__self__",
        None,
    ) and getattr(left, "__func__", left) is getattr(right, "__func__", right)


def _identity_self_attention() -> torch.nn.Module:
    config = replace(
        build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
            return_attention_weights_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        ),
        batch_first_flag=True,
    )
    attention = config.build()
    with torch.no_grad():
        for projection in (
            attention.projector.query_model,
            attention.projector.key_model,
            attention.projector.value_model,
            attention.projector.output_model,
        ):
            layer = projection.layers[0].model
            layer.weight_params.copy_(torch.eye(2))
            layer.bias_params.zero_()
    return attention


def _one_head_mixture_attention() -> torch.nn.Module:
    config = replace(
        build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
            dropout_probability=0.25,
            use_kv_expert_models_flag=False,
            experts_top_k=1,
            experts_num_experts=2,
            experts_compute_expert_mixture_flag=False,
            experts_stack_num_layers=1,
        ),
        batch_first_flag=True,
    )
    return config.build()


def _identity_independent_attention() -> torch.nn.Module:
    config = replace(
        build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
        ),
        batch_first_flag=True,
    )
    attention = config.build()
    with torch.no_grad():
        for projection in (
            attention.projector.query_model,
            attention.projector.key_model,
            attention.projector.value_model,
            attention.projector.output_model,
        ):
            layer = projection.layers[0].model
            layer.weight_params.copy_(torch.eye(2))
            layer.bias_params.zero_()
    return attention


class _AttentionTrainingModule(LightningModule):
    def __init__(
        self,
        *,
        attention: torch.nn.Module | None = None,
        attention_mask: Tensor | None = None,
        fail_after_forward: bool = False,
        learning_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = attention or _identity_self_attention()
        self.register_buffer("attention_mask", attention_mask)
        self.fail_after_forward = fail_after_forward
        self.learning_rate = learning_rate
        self.output: Tensor | None = None
        self.weights: Tensor | None = None

    def training_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        (inputs,) = batch
        output, weights, auxiliary_loss = self.attention(
            inputs,
            inputs,
            inputs,
            attention_mask=self.attention_mask,
        )
        self.output = output.detach().clone()
        self.weights = weights.detach().clone() if weights is not None else None
        if self.fail_after_forward:
            raise RuntimeError("deliberate attention lifecycle failure")
        loss = output.square().mean()
        return loss if auxiliary_loss is None else loss + auxiliary_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


class AttentionMonitorLifecycleTests(unittest.TestCase):
    def test_real_tensorboard_lifecycle_emits_visuals_at_exact_cadence(
        self,
    ) -> None:
        model = _AttentionTrainingModule()
        monitor = AttentionMonitorCallback(
            log_every_n_steps=2,
            history_size=2,
        )
        inputs = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[2.0, 0.0], [0.0, -1.0]],
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            tensorboard_logger = TensorBoardLogger(
                save_dir=temporary_directory,
                name="attention",
            )
            trainer = Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                deterministic=True,
                callbacks=[monitor],
                logger=tensorboard_logger,
                default_root_dir=Path(temporary_directory),
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )

            trainer.fit(
                model,
                train_dataloaders=DataLoader(
                    TensorDataset(inputs),
                    batch_size=1,
                    shuffle=False,
                ),
            )
            tensorboard_logger.experiment.flush()
            events = EventAccumulator(
                tensorboard_logger.log_dir,
                size_guidance={"histograms": 0, "images": 0},
            )
            events.Reload()
            tags = events.Tags()

            histogram_tags = (
                "attention/attention/histogram/entropy_by_head",
                "attention/attention/histogram/max_probability_by_head",
            )
            image_tags = (
                "attention/attention/heatmap/entropy_by_head",
                "attention/attention/heatmap/max_probability_by_head",
            )
            for tag in histogram_tags:
                self.assertIn(tag, tags["histograms"])
                histogram_events = events.Histograms(tag)
                self.assertEqual(
                    [event.step for event in histogram_events],
                    [0, 2],
                )
                for event in histogram_events:
                    self.assertEqual(event.histogram_value.num, 1.0)
                    self.assertTrue(math.isfinite(event.histogram_value.min))
                    self.assertTrue(math.isfinite(event.histogram_value.max))
            for tag in image_tags:
                self.assertIn(tag, tags["images"])
                image_events = events.Images(tag)
                self.assertEqual(
                    [event.step for event in image_events],
                    [0, 2],
                )
                for event in image_events:
                    self.assertGreater(event.width, 0)
                    self.assertGreater(event.height, 0)
                    self.assertTrue(
                        event.encoded_image_string.startswith(b"\x89PNG\r\n\x1a\n")
                    )

        self.assertEqual(monitor._tracker_manager.hook_count, 0)
        self.assertEqual(monitor._tracker_manager.replacement_count, 0)
        self.assertEqual(monitor._entropy_history, {})
        self.assertEqual(monitor._max_probability_history, {})

    def test_fully_masked_independent_lifecycle_is_finite_and_updates_parameters(
        self,
    ) -> None:
        attention_mask = torch.tensor(
            [
                [True, True],
                [False, False],
            ]
        )
        model = _AttentionTrainingModule(
            attention=_identity_independent_attention(),
            attention_mask=attention_mask,
            learning_rate=0.1,
        )
        monitor = AttentionMonitorCallback(log_every_n_steps=1)
        query_layer = model.attention.projector.query_model.layers[0].model
        initial_query_weight = query_layer.weight_params.detach().clone()
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
        inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(inputs),
                batch_size=1,
                shuffle=False,
            ),
        )

        selected_probability = math.exp(2**-0.5) / (1.0 + math.exp(2**-0.5))
        unselected_probability = 1.0 - selected_probability
        expected_output = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [unselected_probability, selected_probability],
                ]
            ]
        )
        entropy = -(
            selected_probability * math.log(selected_probability)
            + unselected_probability * math.log(unselected_probability)
        )
        torch.testing.assert_close(model.output, expected_output)
        self.assertTrue(torch.isfinite(model.output).all())
        torch.testing.assert_close(model.output[:, 0], torch.zeros(1, 2))
        self.assertIsNone(model.weights)
        expected_metrics = {
            "attention/attention/mask_coverage": 0.5,
            "attention/attention/approximate_entropy_mean": entropy / 2,
            "attention/attention/approximate_max_probability_mean": (
                selected_probability / 2
            ),
        }
        for metric_name, expected_value in expected_metrics.items():
            self.assertIn(metric_name, trainer.logged_metrics)
            metric = trainer.logged_metrics[metric_name]
            self.assertTrue(torch.isfinite(metric).item(), metric_name)
            self.assertAlmostEqual(
                metric.item(),
                expected_value,
                places=6,
                msg=metric_name,
            )
        self.assertFalse(
            torch.equal(
                query_layer.weight_params.detach(),
                initial_query_weight,
            )
        )
        self.assertTrue(torch.isfinite(query_layer.weight_params).all())
        self.assertEqual(monitor._tracker_manager.hook_count, 0)
        self.assertEqual(monitor._tracker_manager.replacement_count, 0)
        self.assertEqual(monitor._entropy_history, {})
        self.assertEqual(monitor._max_probability_history, {})

    def test_real_mixture_lifecycle_preserves_one_head_statistics_and_dropout(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = _AttentionTrainingModule(
            attention=_one_head_mixture_attention(),
        )
        monitor = AttentionMonitorCallback(log_every_n_steps=1)
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
                TensorDataset(torch.tensor([[[1.0, -1.0]]])),
                batch_size=1,
                shuffle=False,
            ),
        )

        expected_metrics = {
            "attention/attention/configured_dropout_probability": 0.25,
            "attention/attention/entropy_mean": 0.0,
            "attention/attention/max_probability_mean": 1.0,
            "attention/attention/dead_head_fraction": 1.0,
        }
        for metric_name, expected_value in expected_metrics.items():
            self.assertIn(metric_name, trainer.logged_metrics)
            metric = trainer.logged_metrics[metric_name]
            self.assertTrue(torch.isfinite(metric).item(), metric_name)
            self.assertAlmostEqual(
                metric.item(),
                expected_value,
                places=6,
                msg=metric_name,
            )

    def test_real_trainer_logs_exact_attention_metrics_and_restores_methods(
        self,
    ) -> None:
        model = _AttentionTrainingModule()
        monitor = AttentionMonitorCallback(log_every_n_steps=1)
        original_projection = model.attention.projector.compute_qkv_projections
        original_attention = model.attention.processor.compute_attention
        original_exact_weights = model.attention.processor._SelfAttentionProcessor__compute_masked_attention_weights
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
        inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        trainer.fit(
            model,
            train_dataloaders=DataLoader(
                TensorDataset(inputs),
                batch_size=1,
                shuffle=False,
            ),
        )

        probability = math.exp(2**-0.5) / (math.exp(2**-0.5) + 1.0)
        entropy = -(
            probability * math.log(probability)
            + (1.0 - probability) * math.log(1.0 - probability)
        )
        expected_weights = torch.tensor(
            [
                [
                    [
                        [probability, 1.0 - probability],
                        [1.0 - probability, probability],
                    ]
                ]
            ]
        )
        torch.testing.assert_close(model.weights, expected_weights)
        torch.testing.assert_close(model.output, expected_weights.squeeze(1))
        expected_metrics = {
            "attention/attention/q_norm_mean": 1.0,
            "attention/attention/k_norm_mean": 1.0,
            "attention/attention/v_norm_mean": 1.0,
            "attention/attention/output_norm": math.sqrt(
                2.0 * (probability**2 + (1.0 - probability) ** 2)
            ),
            "attention/attention/configured_dropout_probability": 0.0,
            "attention/attention/mask_coverage": 0.0,
            "attention/attention/entropy_mean": entropy,
            "attention/attention/max_probability_mean": probability,
            "attention/attention/dead_head_fraction": 0.0,
            "attention/attention/dropout_zero_fraction": 0.0,
        }
        self.assertEqual(set(trainer.logged_metrics), set(expected_metrics))
        for metric_name, expected_value in expected_metrics.items():
            self.assertAlmostEqual(
                trainer.logged_metrics[metric_name].item(),
                expected_value,
                places=6,
                msg=metric_name,
            )

        self.assertEqual(monitor._tracker_manager.hook_count, 0)
        self.assertEqual(monitor._tracker_manager.replacement_count, 0)
        self.assertEqual(monitor._entropy_history, {})
        self.assertEqual(monitor._max_probability_history, {})
        self.assertTrue(
            _same_bound_method(
                model.attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertTrue(
            _same_bound_method(
                model.attention.processor.compute_attention,
                original_attention,
            )
        )
        self.assertTrue(
            _same_bound_method(
                model.attention.processor._SelfAttentionProcessor__compute_masked_attention_weights,
                original_exact_weights,
            )
        )

    def test_real_trainer_exception_restores_instrumented_attention_methods(
        self,
    ) -> None:
        model = _AttentionTrainingModule(fail_after_forward=True)
        monitor = AttentionMonitorCallback(log_every_n_steps=1)
        original_projection = model.attention.projector.compute_qkv_projections
        original_attention = model.attention.processor.compute_attention
        original_exact_weights = model.attention.processor._SelfAttentionProcessor__compute_masked_attention_weights
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

        with self.assertRaisesRegex(
            RuntimeError,
            "deliberate attention lifecycle failure",
        ):
            trainer.fit(
                model,
                train_dataloaders=DataLoader(
                    TensorDataset(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])),
                    batch_size=1,
                ),
            )

        self.assertEqual(monitor._tracker_manager.hook_count, 0)
        self.assertEqual(monitor._tracker_manager.replacement_count, 0)
        self.assertEqual(monitor._entropy_history, {})
        self.assertEqual(monitor._max_probability_history, {})
        self.assertTrue(
            _same_bound_method(
                model.attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertTrue(
            _same_bound_method(
                model.attention.processor.compute_attention,
                original_attention,
            )
        )
        self.assertTrue(
            _same_bound_method(
                model.attention.processor._SelfAttentionProcessor__compute_masked_attention_weights,
                original_exact_weights,
            )
        )


if __name__ == "__main__":
    unittest.main()
