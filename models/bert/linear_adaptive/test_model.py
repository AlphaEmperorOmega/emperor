import importlib
import unittest

import torch

from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.bert.linear_adaptive.model import Model
from models.bert.linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
import models.bert.linear_adaptive.config as config
from models.catalog import catalog_entry
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)


import models.bert.linear_adaptive.dataset_options as dataset_options
class TestBertLinearAdaptiveModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.bert.linear_adaptive.config",
            "models.bert.linear_adaptive.presets",
            "models.bert.linear_adaptive.model",
            "models.bert.linear_adaptive.config_builder",
            "models.bert.linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(Experiment()._public_model_id(), "bert/linear_adaptive")
        self.assertIsNotNone(catalog_entry("bert/linear_adaptive"))

    def test_low_rank_preset_uses_adaptive_projection_and_feed_forward_layers(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_WEIGHT)
        encoder_layer_config = self._encoder_layer_config(cfg)

        projection_layer_config = (
            encoder_layer_config.attention_config.projection_model_config.layer_config.layer_model_config
        )
        feed_forward_layer_config = (
            encoder_layer_config.feed_forward_config.stack_config.layer_config.layer_model_config
        )

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
                mlm_logits, nsp_logits, auxiliary_loss = Model(cfg)(
                    *self._fake_bert_inputs(cfg)
                )

                self.assertEqual(mlm_logits.shape, (2, cfg.sequence_length, cfg.output_dim))
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_baseline_trains_one_tiny_epoch(self):
        cfg = self._config(ExperimentPreset.BASELINE)
        model = Model(cfg)
        datamodule = RandomBertPretrainingDataModule(cfg, batch_size=2)

        tiny_cpu_trainer().fit(model, datamodule=datamodule)

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
            "sequence_length": 8,
            "stack_num_layers": 2,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "recurrent_max_steps": 2,
        }

    def _fake_bert_inputs(self, cfg):
        input_ids = torch.randint(5, cfg.input_dim, (2, cfg.sequence_length))
        input_ids[:, 0] = 2
        input_ids[:, -1] = 0
        attention_mask = (input_ids != 0).long()
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, cfg.sequence_length // 2 : -1] = 1
        return input_ids, attention_mask, token_type_ids

    def _encoder_layer_config(self, cfg):
        encoder_config = cfg.experiment_config.encoder_config
        if hasattr(encoder_config, "block_config"):
            encoder_config = encoder_config.block_config
        return encoder_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()
