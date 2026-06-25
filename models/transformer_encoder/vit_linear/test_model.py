import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import models.transformer_encoder.vit_linear.config as config

from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import GridSearch, PresetLock
from emperor.experiments.classifier import ClassifierExperiment
from emperor.transformer import TransformerEncoderBlockLayer, TransformerEncoderLayer
from models.transformer_encoder.vit_linear.config_builder import VitLinearConfigBuilder
from models.transformer_encoder.vit_linear.model import Model
from models.transformer_encoder.vit_linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestVitLinearModel(unittest.TestCase):
    def test_public_imports_remain_available(self):
        for module_name in (
            "models.transformer_encoder.vit_linear.config",
            "models.transformer_encoder.vit_linear.presets",
            "models.transformer_encoder.vit_linear.model",
            "models.transformer_encoder.vit_linear.config_builder",
            "models.transformer_encoder.vit_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(
            Experiment()._public_model_id(),
            "transformer_encoder/vit_linear",
        )

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["vit_linear", "--preset", "baseline"]),
            patch(
                "models.transformer_encoder.vit_linear.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module(
                "models.transformer_encoder.vit_linear.__main__",
                run_name="__main__",
            )

        train_model.assert_called_once()
        experiment = train_model.call_args.args[0]
        kwargs = train_model.call_args.kwargs

        self.assertEqual(experiment.preset, ExperimentPreset.BASELINE)
        self.assertIsNone(kwargs["search_mode"])
        self.assertIsNone(kwargs["log_folder"])
        self.assertIsNone(kwargs["search_keys"])
        self.assertEqual(kwargs["config_overrides"], {})
        self.assertEqual(kwargs["search_overrides"], {})
        self.assertEqual(kwargs["selected_datasets"], config.DATASET_OPTIONS)
        self.assertIsNone(kwargs["selected_presets"])

    def test_modern_preset_contract_is_exposed(self):
        expected_overrides = {
            ExperimentPreset.BASELINE: {},
            ExperimentPreset.POST_NORM: {
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            },
            ExperimentPreset.SINUSOIDAL: {
                "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
            },
            ExperimentPreset.ATTENTION_BIAS: {
                "attn_bias_flag": True,
                "attn_add_key_value_bias_flag": True,
            },
        }

        self.assertEqual(ExperimentPresets.PRESET_OVERRIDES, expected_overrides)
        for preset, overrides in expected_overrides.items():
            if not overrides:
                continue
            with self.subTest(preset=preset.name):
                self.assertEqual(
                    {
                        key: lock.value
                        for key, lock in ExperimentPresets.PRESET_LOCKS[
                            preset
                        ].items()
                    },
                    overrides,
                )

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = ExperimentPresets()

        for preset, expected_locks in presets.PRESET_LOCKS.items():
            with self.subTest(preset=preset.name):
                locks = presets.locked_fields(preset)

                self.assertEqual(set(locks), set(expected_locks))
                for field, lock in locks.items():
                    expected = expected_locks[field]
                    expected_value = (
                        expected.value if isinstance(expected, PresetLock) else expected
                    )
                    self.assertEqual(lock.value, expected_value)
                    self.assertIn(preset.name, lock.reason)

    def test_all_presets_forward_one_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    dataset,
                    config_overrides=self._test_overrides(batch_size),
                )[0]
                model = Model(cfg)
                images = self._fake_batch(dataset, batch_size)

                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_baseline_forwards_all_image_datasets(self):
        batch_size = 2
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides=self._test_overrides(batch_size),
                )[0]
                model = Model(cfg)
                images = self._fake_batch(dataset, batch_size)

                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_all_presets_train_one_epoch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    dataset,
                    config_overrides=self._test_overrides(batch_size),
                )[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(
                    dataset,
                    batch_size=batch_size,
                )

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_search_applies_stack_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["stack_num_layers"],
            config_overrides=self._test_overrides(batch_size=2),
        )

        self.assertEqual(
            len(configs),
            len(config.SEARCH_SPACE_STACK_NUM_LAYERS),
        )
        self.assertEqual(
            {
                cfg.experiment_config.encoder_config.num_layers
                for cfg in configs
            },
            set(config.SEARCH_SPACE_STACK_NUM_LAYERS),
        )

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.BASELINE,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_keys=["bogus_axis"],
                config_overrides=self._test_overrides(batch_size=2),
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_unlocked_overrides_update_flat_and_nested_config(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides={
                "batch_size": 2,
                "stack_hidden_dim": 24,
                "stack_num_layers": 2,
                "stack_activation": ActivationOptions.RELU,
                "stack_dropout_probability": 0.2,
                "patch_dropout_probability": 0.3,
                "attn_num_heads": 4,
                "output_num_layers": 1,
            },
        )[0]

        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(cfg.experiment_config.encoder_config.num_layers, 2)
        self.assertEqual(self._encoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(self._attention_config(cfg).dropout_probability, 0.2)
        self.assertEqual(cfg.experiment_config.patch_config.dropout_probability, 0.3)
        self.assertEqual(cfg.experiment_config.output_config.num_layers, 1)
        self.assertEqual(
            cfg.experiment_config.output_config.layer_config.activation,
            ActivationOptions.RELU,
        )

    def test_locked_preset_rejects_conflicting_overrides(self):
        presets = ExperimentPresets()

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.POST_NORM,
                config.DATASET_OPTIONS[0],
                config_overrides={
                    **self._test_overrides(batch_size=2),
                    "layer_norm_position": LayerNormPositionOptions.BEFORE,
                },
            )

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.POST_NORM,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_keys=["layer_norm_position"],
                config_overrides=self._test_overrides(batch_size=2),
            )

        with self.assertRaisesRegex(ValueError, "POST_NORM.*layer_norm_position"):
            presets.get_config(
                ExperimentPreset.POST_NORM,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_overrides={
                    "layer_norm_position": [LayerNormPositionOptions.BEFORE],
                },
                config_overrides=self._test_overrides(batch_size=2),
            )

    def test_model_inherits_classifier_experiment(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        self.assertIsInstance(model, ClassifierExperiment)
        self.assertIsInstance(model.encoder_layer_norm, nn.LayerNorm)
        self.assertEqual(
            model.encoder_layer_norm.normalized_shape,
            (cfg.hidden_dim,),
        )

    def test_patch_embedding_prepends_class_token(self):
        batch_size = 2
        dataset = config.DATASET_OPTIONS[0]
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset,
            config_overrides=self._test_overrides(batch_size),
        )[0]
        model = Model(cfg)
        images = self._fake_batch(dataset, batch_size)

        patch_embeddings = model.patch(images)

        expected_sequence_length = self._expected_sequence_length(
            dataset.default_height,
            cfg.experiment_config.patch_config.patch_size,
        )
        self.assertEqual(
            patch_embeddings.shape,
            (batch_size, expected_sequence_length, cfg.hidden_dim),
        )
        expected_class_token = model.patch.class_token.expand(batch_size, -1, -1)
        torch.testing.assert_close(
            patch_embeddings[:, :1, :],
            expected_class_token,
        )

    def test_class_token_positional_embedding_is_trainable(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        embedding = model.positional_embedding.embedding_model

        self.assertIsNone(embedding.padding_idx)
        self.assertTrue(embedding.weight.requires_grad)

    def test_encoder_is_built_from_transformer_encoder_block_layers(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        encoder_layers = self._encoder_layers(model)

        self.assertGreater(len(encoder_layers), 0)
        for layer in encoder_layers:
            self.assertIsInstance(layer, TransformerEncoderBlockLayer)
            self.assertIsInstance(layer.model, TransformerEncoderLayer)

    def test_model_step_accepts_classifier_batch(self):
        batch_size = 2
        dataset = config.DATASET_OPTIONS[0]
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset,
            config_overrides=self._test_overrides(batch_size),
        )[0]
        model = Model(cfg)
        images = self._fake_batch(dataset, batch_size)
        labels = torch.randint(0, dataset.num_classes, (batch_size,))

        loss, logits, returned_labels = model._model_step((images, labels))

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.shape, (batch_size, dataset.num_classes))
        torch.testing.assert_close(returned_labels, labels)

    def test_auxiliary_loss_from_encoder_is_included_by_classifier_experiment(self):
        batch_size = 2
        dataset = config.DATASET_OPTIONS[0]
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset,
            config_overrides=self._test_overrides(batch_size),
        )[0]
        model = Model(cfg)
        model.transformer = _AuxiliaryLossEncoder(0.25)
        images = self._fake_batch(dataset, batch_size)
        labels = torch.randint(0, dataset.num_classes, (batch_size,))

        loss, logits, _labels = model._model_step((images, labels))

        expected_loss = model.loss_fn(logits, labels) + logits.new_tensor(0.25)
        torch.testing.assert_close(loss, expected_loss)

    def test_dataset_metadata_drives_channels_classes_and_sequence_length(self):
        presets = ExperimentPresets()

        for dataset in config.DATASET_OPTIONS:
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides=self._test_overrides(batch_size=2),
                )[0]
                patch_cfg = cfg.experiment_config.patch_config
                positional_cfg = cfg.experiment_config.positional_embedding_config
                attention_cfg = self._attention_config(cfg)
                expected_sequence_length = self._expected_sequence_length(
                    dataset.default_height,
                    patch_cfg.patch_size,
                )

                self.assertEqual(patch_cfg.num_input_channels, dataset.num_channels)
                self.assertEqual(cfg.output_dim, dataset.num_classes)
                self.assertEqual(cfg.sequence_length, expected_sequence_length)
                self.assertEqual(positional_cfg.num_embeddings, expected_sequence_length - 1)
                self.assertTrue(positional_cfg.class_token_flag)
                self.assertEqual(attention_cfg.target_sequence_length, expected_sequence_length)
                self.assertEqual(attention_cfg.source_sequence_length, expected_sequence_length)

    def test_presets_wire_config_variants(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentPreset.BASELINE)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            ImageLearnedPositionalEmbeddingConfig,
        )
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(self._encoder_layer_config(cfg).causal_attention_mask_flag)
        self.assertFalse(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentPreset.POST_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

        cfg = presets.get_config(ExperimentPreset.SINUSOIDAL)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            ImageSinusoidalPositionalEmbeddingConfig,
        )

        cfg = presets.get_config(ExperimentPreset.ATTENTION_BIAS)[0]
        attention_cfg = self._attention_config(cfg)
        self.assertTrue(attention_cfg.add_key_value_bias_flag)
        self.assertTrue(
            attention_cfg.projection_model_config.layer_config.layer_model_config.bias_flag
        )

    def test_invalid_patch_size_for_image_height_raises(self):
        with self.assertRaises(ValueError):
            VitLinearConfigBuilder(image_height=30, image_patch_size=4).build()

    def _test_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "stack_hidden_dim": 16,
            "stack_num_layers": 1,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "output_num_layers": 1,
        }

    def _fake_batch(self, dataset: type, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            dataset.num_channels,
            dataset.default_height,
            dataset.default_width,
        )

    def _expected_sequence_length(self, image_height: int, patch_size: int) -> int:
        patches_per_axis = image_height // patch_size
        return patches_per_axis * patches_per_axis + 1

    def _encoder_layers(self, model) -> list:
        transformer = model.transformer
        if isinstance(transformer, nn.Sequential) or hasattr(transformer, "layers"):
            return list(transformer)
        return [transformer]

    def _encoder_config(self, cfg):
        return cfg.experiment_config.encoder_config

    def _encoder_layer_config(self, cfg):
        return self._encoder_config(cfg).layer_config.layer_model_config

    def _attention_config(self, cfg):
        return self._encoder_layer_config(cfg).attention_config


class _AuxiliaryLossEncoder(nn.Module):
    def __init__(self, auxiliary_loss: float):
        super().__init__()
        self.auxiliary_loss = auxiliary_loss

    def forward(self, state):
        state.loss = state.hidden.new_tensor(self.auxiliary_loss)
        return state


if __name__ == "__main__":
    unittest.main()
