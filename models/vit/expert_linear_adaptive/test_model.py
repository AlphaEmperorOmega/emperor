import importlib
import unittest

import torch

from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.catalog import catalog_entry
import models.vit.expert_linear_adaptive.config as config
from models.vit.expert_linear_adaptive.model import Model
from models.vit.expert_linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)


class TestVitExpertLinearAdaptiveModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.vit.expert_linear_adaptive.config",
            "models.vit.expert_linear_adaptive.presets",
            "models.vit.expert_linear_adaptive.model",
            "models.vit.expert_linear_adaptive.config_builder",
            "models.vit.expert_linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(
            Experiment()._public_model_id(),
            "vit/expert_linear_adaptive",
        )
        self.assertIsNotNone(catalog_entry("vit/expert_linear_adaptive"))

    def test_low_rank_expert_preset_uses_adaptive_expert_layers(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_EXPERT_WEIGHT)
        feed_forward_stack_config = (
            self._encoder_layer_config(cfg).feed_forward_config.stack_config
        )
        expert_layer_config = (
            feed_forward_stack_config.stack_config.layer_config.layer_model_config
            .expert_model_config.layer_config.layer_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertIsInstance(expert_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            expert_layer_config.adaptive_augmentation_config.weight_config
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
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(),
        )[0]

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "stack_hidden_dim": 16,
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
