import importlib
import unittest

import torch

from emperor.attention.core.variants.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.linears.core.config import LinearLayerConfig
from models.catalog import catalog_entry
import models.vit.expert_linear.config as config
from models.vit.expert_linear.model import Model
from models.vit.expert_linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)


import models.vit.expert_linear.dataset_options as dataset_options
class TestVitExpertLinearModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.vit.expert_linear.config",
            "models.vit.expert_linear.presets",
            "models.vit.expert_linear.model",
            "models.vit.expert_linear.config_builder",
            "models.vit.expert_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(Experiment()._public_model_id(), "vit/expert_linear")
        self.assertIsNotNone(catalog_entry("vit/expert_linear"))

    def test_feed_forward_stack_is_expert_backed(self):
        cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
        feed_forward_stack_config = (
            self._encoder_layer_config(cfg).feed_forward_config.stack_config
        )
        expert_core_config = (
            feed_forward_stack_config.stack_config.layer_config.layer_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertEqual(feed_forward_stack_config.top_k, 1)
        self.assertEqual(feed_forward_stack_config.sampler_config.switch_loss_weight, 0.1)
        self.assertIsInstance(
            expert_core_config.expert_model_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_expert_attention_preset_uses_mixture_of_attention_heads(self):
        cfg = self._config(ExperimentPreset.EXPERT_ATTENTION)
        attention_config = self._encoder_layer_config(cfg).attention_config

        self.assertIsInstance(attention_config, MixtureOfAttentionHeadsConfig)
        self.assertFalse(attention_config.use_kv_expert_models_flag)

    def test_baseline_attention_remains_self_attention(self):
        cfg = self._config(ExperimentPreset.BASELINE)

        self.assertIsInstance(
            self._encoder_layer_config(cfg).attention_config,
            SelfAttentionConfig,
        )

    def test_all_presets_forward_one_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                output = Model(cfg)(self._fake_batch(cfg))
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (2, cfg.output_dim))

    def _config(self, preset: ExperimentPreset):
        return ExperimentPresets().get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[dataset_options.DEFAULT_EXPERIMENT_TASK][0],
            config_overrides=self._test_overrides(),
        )[0]

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "output_num_layers": 1,
        }

    def _fake_batch(self, cfg):
        patch_config = cfg.experiment_config.patch_config
        return torch.randn(
            2,
            patch_config.num_input_channels,
            patch_config.patch_size * 7,
            patch_config.patch_size * 7,
        )

    def _encoder_layer_config(self, cfg):
        return cfg.experiment_config.encoder_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()
