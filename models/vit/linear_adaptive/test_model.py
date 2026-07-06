import importlib
import unittest

import torch

from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from models.catalog import catalog_entry
import models.vit.linear_adaptive.config as config
from models.vit.linear_adaptive.model import Model
from models.vit.linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)


class TestVitLinearAdaptiveModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.vit.linear_adaptive.config",
            "models.vit.linear_adaptive.presets",
            "models.vit.linear_adaptive.model",
            "models.vit.linear_adaptive.config_builder",
            "models.vit.linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(Experiment()._public_model_id(), "vit/linear_adaptive")
        self.assertIsNotNone(catalog_entry("vit/linear_adaptive"))

    def test_low_rank_preset_adapts_only_encoder_backend_stacks(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_WEIGHT)
        experiment_config = cfg.experiment_config
        encoder_layer_config = self._encoder_layer_config(cfg)

        patch_layer_config = (
            experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config
        )
        output_layer_config = (
            experiment_config.output_config.layer_config.layer_model_config
        )
        projection_layer_config = (
            encoder_layer_config.attention_config.projection_model_config.layer_config.layer_model_config
        )
        feed_forward_layer_config = (
            encoder_layer_config.feed_forward_config.stack_config.layer_config.layer_model_config
        )

        self.assertIsInstance(patch_layer_config, LinearLayerConfig)
        self.assertIsInstance(output_layer_config, LinearLayerConfig)
        self.assertIsInstance(projection_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsInstance(feed_forward_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
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
