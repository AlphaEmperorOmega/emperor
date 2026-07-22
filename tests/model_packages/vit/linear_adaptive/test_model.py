# ruff: noqa: E501

import importlib
import unittest
from pathlib import Path

import torch

import models.vit.linear_adaptive.config as config
import models.vit.linear_adaptive.dataset_options as dataset_options
from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.layers import LayerConfig
from emperor.linears import LinearLayerConfig
from models.catalog import model_package
from models.vit.linear_adaptive.config_builder import VitLinearAdaptiveConfigBuilder
from models.vit.linear_adaptive.model import Model
from models.vit.linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
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

        experiment = Experiment(model_package=model_package("vit/linear_adaptive"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "vit/linear_adaptive",
        )
        self.assertIsNotNone(model_package("vit/linear_adaptive"))

    def test_config_builder_uses_local_adaptive_defaults(self):
        local_hidden_dim = config.HIDDEN_DIM + 11
        runtime = model_package("vit/linear_adaptive").bind_runtime_defaults(
            {"adaptive_generator_stack_hidden_dim": local_hidden_dim}
        )
        builder = VitLinearAdaptiveConfigBuilder(runtime=runtime)

        self.assertEqual(
            builder.adaptive_generator_stack_options.hidden_dim,
            local_hidden_dim,
        )

    def test_package_does_not_import_generic_linear_adaptive_config(self):
        package_dir = Path(config.__file__).resolve().parent

        for file_name in ("config.py", "config_builder.py", "presets.py"):
            with self.subTest(file_name=file_name):
                source = (package_dir / file_name).read_text(encoding="utf-8")
                self.assertNotIn(
                    "models.linears.linear_adaptive" + ".config",
                    source,
                )

    def test_low_rank_preset_adapts_only_encoder_backend_stacks(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_WEIGHT)
        experiment_config = cfg.experiment_config
        encoder_layer_config = self._encoder_layer_config(cfg)

        patch_layer_config = experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config
        output_layer_config = experiment_config.output_config.layer_model_config
        projection_layer_config = encoder_layer_config.attention_config.projection_model_config.layer_config.layer_model_config
        feed_forward_layer_config = encoder_layer_config.feed_forward_config.stack_config.layer_config.layer_model_config

        self.assertIsInstance(patch_layer_config, LinearLayerConfig)
        self.assertIsInstance(experiment_config.output_config, LayerConfig)
        self.assertIsInstance(output_layer_config, LinearLayerConfig)
        self.assertIsInstance(projection_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsInstance(feed_forward_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_global_weight_override_adapts_attention_and_feed_forward(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
            },
        )

        projection_layer_config = self._projection_layer_config(cfg)
        feed_forward_layer_config = self._feed_forward_layer_config(cfg)

        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_attention_weight_override_does_not_enable_feed_forward_weight(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "attn_weight_option_flag": True,
                "attn_weight_option": config.LowRankDynamicWeightConfig,
            },
        )

        projection_layer_config = self._projection_layer_config(cfg)
        feed_forward_layer_config = self._feed_forward_layer_config(cfg)

        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_feed_forward_weight_override_does_not_enable_attention_weight(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "ff_weight_option_flag": True,
                "ff_weight_option": config.LowRankDynamicWeightConfig,
            },
        )

        projection_layer_config = self._projection_layer_config(cfg)
        feed_forward_layer_config = self._feed_forward_layer_config(cfg)

        self.assertIsNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_feed_forward_weight_override_can_disable_global_weight(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
                "ff_weight_option_flag": False,
            },
        )

        projection_layer_config = self._projection_layer_config(cfg)
        feed_forward_layer_config = self._feed_forward_layer_config(cfg)

        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_role_generator_stack_overrides_are_independent(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
                "attn_weight_generator_stack_independent_flag": True,
                "attn_weight_generator_stack_hidden_dim": 23,
                "ff_weight_generator_stack_independent_flag": True,
                "ff_weight_generator_stack_hidden_dim": 37,
            },
        )

        projection_weight_config = self._projection_layer_config(
            cfg
        ).adaptive_augmentation_config.weight_config
        feed_forward_weight_config = self._feed_forward_layer_config(
            cfg
        ).adaptive_augmentation_config.weight_config

        self.assertEqual(projection_weight_config.model_config.hidden_dim, 23)
        self.assertEqual(feed_forward_weight_config.model_config.hidden_dim, 37)

    def test_patch_embedding_and_output_head_remain_plain_with_role_overrides(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            config_overrides={
                **self._test_overrides(),
                "attn_weight_option_flag": True,
                "attn_weight_option": config.LowRankDynamicWeightConfig,
                "ff_weight_option_flag": True,
                "ff_weight_option": config.LowRankDynamicWeightConfig,
            },
        )
        experiment_config = cfg.experiment_config
        patch_layer_config = experiment_config.patch_config.embedding_stack_config.layer_config.layer_model_config
        output_layer_config = experiment_config.output_config.layer_model_config

        self.assertIsInstance(patch_layer_config, LinearLayerConfig)
        self.assertIsInstance(experiment_config.output_config, LayerConfig)
        self.assertIsInstance(output_layer_config, LinearLayerConfig)

    def test_feed_forward_stack_controls_keep_adaptive_layers_inside_ff_stack(self):
        cfg = self._config(
            ExperimentPreset.LOW_RANK_WEIGHT,
            config_overrides={
                **self._test_overrides(),
                "ff_stack_hidden_dim": 17,
                "ff_stack_gate_flag": True,
            },
        )
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        feed_forward_layer_config = (
            feed_forward_stack_config.layer_config.layer_model_config
        )

        self.assertEqual(feed_forward_stack_config.hidden_dim, 17)
        self.assertIsNotNone(feed_forward_stack_config.layer_config.gate_config)
        self.assertIsInstance(feed_forward_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_attention_projection_controls_keep_adaptive_layers(self):
        cfg = self._config(
            ExperimentPreset.LOW_RANK_WEIGHT,
            config_overrides={
                **self._test_overrides(),
                "attn_stack_hidden_dim": 17,
                "attn_stack_gate_flag": True,
            },
        )
        projection_stack_config = self._encoder_layer_config(
            cfg
        ).attention_config.projection_model_config
        projection_layer_config = (
            projection_stack_config.layer_config.layer_model_config
        )

        self.assertEqual(projection_stack_config.hidden_dim, 17)
        self.assertIsNotNone(projection_stack_config.layer_config.gate_config)
        self.assertIsInstance(projection_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_all_presets_forward_one_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                output = Model(cfg)(self._fake_batch(cfg))
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (2, cfg.output_dim))

    def _config(self, preset: ExperimentPreset, config_overrides: dict | None = None):
        return model_package("vit/linear_adaptive").presets.get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides=config_overrides or self._runtime_overrides(),
        )[0]

    def _runtime_overrides(self) -> dict:
        return {"batch_size": 2}

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 4,
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

    def _projection_layer_config(self, cfg):
        return self._encoder_layer_config(
            cfg
        ).attention_config.projection_model_config.layer_config.layer_model_config

    def _feed_forward_layer_config(self, cfg):
        return self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()
