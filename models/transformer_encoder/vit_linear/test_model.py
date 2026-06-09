import unittest

import torch
import torch.nn as nn

import models.transformer_encoder.vit_linear.config as config

from emperor.base.options import LayerNormPositionOptions
from emperor.embedding.absolute.core.config import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.classifier import ClassifierExperiment
from emperor.transformer import TransformerEncoderBlockLayer, TransformerEncoderLayer
from models.transformer_encoder.vit_linear.config_builder import VitLinearConfigBuilder
from models.transformer_encoder.vit_linear.model import Model
from models.transformer_encoder.vit_linear.presets import ExperimentOptions, ExperimentPresets
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)


class TestVitLinearModel(unittest.TestCase):
    def test_all_options_forward_one_batch(self):
        batch_size = 2
        presets = ExperimentPresets()
        dataset = config.DATASET_OPTIONS[0]

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                cfg = presets.get_config(
                    option,
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
                    ExperimentOptions.BASELINE,
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

        for option in ExperimentOptions:
            with self.subTest(option=option.name):
                cfg = presets.get_config(
                    option,
                    dataset,
                    config_overrides=self._test_overrides(batch_size),
                )[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(
                    dataset,
                    batch_size=batch_size,
                )

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_model_inherits_classifier_experiment(self):
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
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
            ExperimentOptions.BASELINE,
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
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        embedding = model.positional_embedding.embedding_model

        self.assertIsNone(embedding.padding_idx)
        self.assertTrue(embedding.weight.requires_grad)

    def test_encoder_is_built_from_transformer_encoder_block_layers(self):
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
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
            ExperimentOptions.BASELINE,
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
            ExperimentOptions.BASELINE,
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
                    ExperimentOptions.BASELINE,
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

        cfg = presets.get_config(ExperimentOptions.BASELINE)[0]
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

        cfg = presets.get_config(ExperimentOptions.POST_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

        cfg = presets.get_config(ExperimentOptions.SINUSOIDAL)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            ImageSinusoidalPositionalEmbeddingConfig,
        )

        cfg = presets.get_config(ExperimentOptions.ATTENTION_BIAS)[0]
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
            "hidden_dim": 16,
            "transformer_num_layers": 1,
            "attn_num_heads": 4,
            "dropout_probability": 0.0,
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
