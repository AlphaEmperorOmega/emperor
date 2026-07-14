import unittest

import torch
from emperor.attention.core.monitor import AttentionMonitorCallback
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig

from support.attention import build_attention_config
from support.monitor import (
    CaptureLightningModule,
    NoExperimentLightningModule,
    TrainerStub,
    same_bound_method,
)


class TestAttentionMonitorCallback(unittest.TestCase):
    def attention(self):
        cfg = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
            return_attention_weights_flag=True,
        )
        return cfg.build()

    def qkv(self):
        tensor = torch.randn(3, 2, 4)
        return tensor, tensor, tensor

    def test_rejects_non_positive_cadence(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    AttentionMonitorCallback(log_every_n_steps=bad)

    def test_discovers_only_attention_modules(self):
        module = CaptureLightningModule(
            attn=self.attention(), other=torch.nn.Linear(4, 4)
        )
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)

        self.assertEqual([name for name, _ in callback._attention_modules], ["attn"])
        callback.on_fit_end(TrainerStub(), module)

    def test_respects_global_step_cadence(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=2)
        callback.on_fit_start(TrainerStub(), module)

        module.global_step = 1
        attention(*self.qkv())
        self.assertEqual(module.logged, [])

        module.global_step = 2
        attention(*self.qkv())
        self.assertIn("attn/attention/q_norm_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_logs_expected_finite_scalars(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        expected_tags = {
            "attn/attention/q_norm_mean",
            "attn/attention/k_norm_mean",
            "attn/attention/v_norm_mean",
            "attn/attention/output_norm",
            "attn/attention/entropy_mean",
            "attn/attention/max_probability_mean",
            "attn/attention/dead_head_fraction",
            "attn/attention/mask_coverage",
            "attn/attention/configured_dropout_probability",
            "attn/attention/dropout_zero_fraction",
        }
        self.assertTrue(expected_tags.issubset(set(module.logged_tags)))
        for tag in expected_tags:
            value = module.logged_value(tag)
            self.assertTrue(torch.isfinite(torch.as_tensor(value)).all(), tag)
        callback.on_fit_end(TrainerStub(), module)

    def test_emits_histograms_and_images_when_experiment_supports_them(self):
        attention = self.attention()
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        experiment = module.logger.experiment
        self.assertTrue(
            any(
                tag == "attn/attention/histogram/entropy_by_head"
                for tag, _, _ in experiment.histograms
            )
        )
        self.assertTrue(
            any(
                tag == "attn/attention/heatmap/entropy_by_head"
                for tag, _, _, _ in experiment.images
            )
        )
        callback.on_fit_end(TrainerStub(), module)

    def test_skips_visual_summaries_without_experiment(self):
        attention = self.attention()
        module = NoExperimentLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)
        callback.on_fit_start(TrainerStub(), module)

        attention(*self.qkv())

        self.assertIn("attn/attention/entropy_mean", module.logged_tags)
        callback.on_fit_end(TrainerStub(), module)

    def test_restores_wrappers_and_clears_state_on_fit_end(self):
        attention = self.attention()
        original_projection = attention.projector.compute_qkv_projections
        original_processor = attention.processor.compute_attention
        module = CaptureLightningModule(attn=attention)
        callback = AttentionMonitorCallback(log_every_n_steps=1)

        callback.on_fit_start(TrainerStub(), module)
        self.assertIsNot(
            attention.projector.compute_qkv_projections, original_projection
        )
        self.assertIsNot(attention.processor.compute_attention, original_processor)

        callback.on_fit_end(TrainerStub(), module)

        self.assertTrue(
            same_bound_method(
                attention.projector.compute_qkv_projections,
                original_projection,
            )
        )
        self.assertTrue(
            same_bound_method(
                attention.processor.compute_attention,
                original_processor,
            )
        )
        self.assertEqual(callback._attention_modules, [])
        self.assertEqual(callback._wrapped_methods, [])
        self.assertEqual(callback._traces, {})


if __name__ == "__main__":
    unittest.main()
