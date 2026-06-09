import unittest

import torch

from torch import nn

import models.transformer_encoder.bert_linear.config as config

from emperor.base.options import LayerNormPositionOptions
from emperor.attention import AttentionLayerState
from emperor.embedding.absolute.core.config import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import RandomSearch
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.transformer import (
    TransformerEncoderBlockLayer,
    TransformerEncoderLayer,
)
from models.transformer_encoder.bert_linear.model import Model
from models.transformer_encoder.bert_linear.presets import ExperimentOptions, ExperimentPresets
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)


class TestBertLinearModel(unittest.TestCase):
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
                batch = self._fake_bert_inputs(cfg, batch_size)

                mlm_logits, nsp_logits, auxiliary_loss = model(*batch)

                self.assertEqual(
                    mlm_logits.shape,
                    (batch_size, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (batch_size, 2))
                self.assertIsNotNone(auxiliary_loss)

    def test_baseline_forwards_all_datasets(self):
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
                batch = self._fake_bert_inputs(cfg, batch_size)

                mlm_logits, nsp_logits, auxiliary_loss = model(*batch)

                self.assertEqual(
                    mlm_logits.shape,
                    (batch_size, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (batch_size, 2))
                self.assertIsNotNone(auxiliary_loss)

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
                datamodule = RandomBertPretrainingDataModule(cfg, batch_size=batch_size)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _test_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "hidden_dim": 16,
            "sequence_length": 8,
            "stack_num_layers": 1,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
        }

    def _fake_bert_inputs(
        self, cfg, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.randint(
            5,
            cfg.input_dim,
            (batch_size, cfg.sequence_length),
        )
        input_ids[:, 0] = 2
        input_ids[:, -1] = 0
        attention_mask = (input_ids != 0).long()
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, cfg.sequence_length // 2 : -1] = 1
        return input_ids, attention_mask, token_type_ids

    def test_preset_accepts_search_flags(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentOptions.BASELINE)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            TextLearnedPositionalEmbeddingConfig,
        )
        self.assertFalse(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentOptions.PRE_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )

        cfg = presets.get_config(ExperimentOptions.POST_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

        cfg = presets.get_config(ExperimentOptions.SINUSOIDAL)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            TextSinusoidalPositionalEmbeddingConfig,
        )

        cfg = presets.get_config(ExperimentOptions.CAUSAL)[0]
        self.assertTrue(self._encoder_layer_config(cfg).causal_attention_mask_flag)
        self.assertTrue(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentOptions.ATTENTION_BIAS)[0]
        attention_config = self._attention_config(cfg)
        self.assertTrue(attention_config.add_key_value_bias_flag)
        self.assertTrue(
            attention_config.projection_model_config.layer_config.layer_model_config.bias_flag
        )

    def test_model_uses_bert_pretraining_base_class_and_heads(self):
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        self.assertIsInstance(model, BertPretrainingExperiment)
        self.assertIsInstance(model.token_type_embedding, nn.Embedding)
        self.assertEqual(model.token_type_embedding.num_embeddings, 2)
        self.assertIsInstance(model.embedding_layer_norm, nn.LayerNorm)
        self.assertEqual(
            model.embedding_layer_norm.normalized_shape,
            (cfg.hidden_dim,),
        )
        self.assertIsInstance(model.encoder_layer_norm, nn.LayerNorm)
        self.assertEqual(
            model.encoder_layer_norm.normalized_shape,
            (cfg.hidden_dim,),
        )
        self.assertIs(model.mlm_decoder.weight, model.token_embedding.weight)
        self.assertEqual(model.nsp_head.out_features, 2)

    def test_model_step_accepts_canonical_bert_pretraining_batch(self):
        batch_size = 2
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size),
        )[0]
        model = Model(cfg)
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(
            cfg, batch_size
        )
        mlm_labels = torch.full_like(input_ids, -100)
        mlm_labels[0, 1] = input_ids[0, 1]
        mlm_labels[1, 2] = input_ids[1, 2]
        next_sentence_labels = torch.tensor([0, 1])

        loss = model._model_step(
            (
                input_ids,
                mlm_labels,
                attention_mask,
                token_type_ids,
                next_sentence_labels,
            )
        )

        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_converts_attention_mask_to_encoder_key_padding_mask(self):
        batch_size = 2
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides=self._test_overrides(batch_size),
        )[0]
        model = Model(cfg)

        class SpyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None
                self.key_padding_mask = None
                self.attention_mask = None

            def forward(self, state):
                self.state = state
                self.key_padding_mask = state.key_padding_mask
                self.attention_mask = state.attention_mask
                state.loss = state.hidden.new_zeros(())
                return state

        spy = SpyEncoder()
        model.transformer = spy
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(
            cfg, batch_size
        )

        model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        self.assertIsInstance(spy.state, AttentionLayerState)
        torch.testing.assert_close(
            spy.key_padding_mask,
            attention_mask == 0,
        )
        self.assertIsNone(spy.attention_mask)

    def test_encoder_built_from_block_layers(self):
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


if __name__ == "__main__":
    unittest.main()
