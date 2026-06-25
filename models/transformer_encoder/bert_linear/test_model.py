import importlib
import runpy
import sys
import unittest
from unittest.mock import patch

import torch

from torch import nn

import models.transformer_encoder.bert_linear.config as config

from emperor.base.options import LayerNormPositionOptions
from emperor.attention import AttentionLayerState
from emperor.embedding.absolute.core.config import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import GridSearch, PresetLock, RandomSearch
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.transformer import (
    TransformerEncoderBlockLayer,
    TransformerEncoderLayer,
)
from models.transformer_encoder.bert_linear.model import Model
from models.transformer_encoder.bert_linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)


class TestBertLinearModel(unittest.TestCase):
    def test_public_imports_remain_available(self):
        for module_name in (
            "models.transformer_encoder.bert_linear.config",
            "models.transformer_encoder.bert_linear.presets",
            "models.transformer_encoder.bert_linear.model",
            "models.transformer_encoder.bert_linear.config_builder",
            "models.transformer_encoder.bert_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(
            Experiment()._public_model_id(),
            "transformer_encoder/bert_linear",
        )

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["bert_linear", "--preset", "baseline"]),
            patch(
                "models.transformer_encoder.bert_linear.presets.Experiment.train_model",
                autospec=True,
            ) as train_model,
        ):
            runpy.run_module(
                "models.transformer_encoder.bert_linear.__main__",
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
            ExperimentPreset.PRE_NORM: {
                "layer_norm_position": LayerNormPositionOptions.BEFORE,
            },
            ExperimentPreset.POST_NORM: {
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            },
            ExperimentPreset.SINUSOIDAL: {
                "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
            },
            ExperimentPreset.CAUSAL: {
                "causal_attention_mask_flag": True,
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

    def test_skipped_controller_stack_config_constants_are_canonical(self):
        canonical_names = {
            "GATE_STACK_HIDDEN_DIM",
            "GATE_STACK_LAYER_NORM_POSITION",
            "GATE_STACK_BIAS_FLAG",
            "HALTING_STACK_HIDDEN_DIM",
            "HALTING_STACK_LAYER_NORM_POSITION",
            "HALTING_STACK_BIAS_FLAG",
        }
        legacy_names = {name.replace("_STACK_", "_") for name in canonical_names}

        for name in canonical_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name))
                self.assertIn(name, config.CONFIG_OVERRIDE_SKIP_KEYS)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(config, name))
                self.assertNotIn(name, config.CONFIG_OVERRIDE_SKIP_KEYS)

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
                    ExperimentPreset.BASELINE,
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

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    dataset,
                    config_overrides=self._test_overrides(batch_size),
                )[0]
                model = Model(cfg)
                datamodule = RandomBertPretrainingDataModule(cfg, batch_size=batch_size)

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _test_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "stack_hidden_dim": 16,
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
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_search_keys_unknown_axis_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ExperimentPresets().get_config(
                ExperimentPreset.BASELINE,
                config.DATASET_OPTIONS[0],
                RandomSearch(num_samples=2),
                search_keys=["bogus_axis"],
            )

        self.assertIn("Unknown", str(ctx.exception))

    def test_search_applies_encoder_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            GridSearch(),
            search_keys=["stack_hidden_dim"],
        )

        self.assertEqual(len(configs), len(config.SEARCH_SPACE_STACK_HIDDEN_DIM))
        self.assertEqual(
            {cfg.hidden_dim for cfg in configs},
            set(config.SEARCH_SPACE_STACK_HIDDEN_DIM),
        )

    def test_unlocked_overrides_update_flat_and_nested_config(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            config.DATASET_OPTIONS[0],
            config_overrides={
                "stack_hidden_dim": 24,
                "sequence_length": 8,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.2,
                "attn_num_heads": 2,
                "ff_num_layers": 1,
            },
        )[0]

        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(cfg.sequence_length, 8)
        self.assertEqual(self._encoder_config(cfg).num_layers, 1)
        self.assertEqual(self._encoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(self._attention_config(cfg).num_heads, 2)
        self.assertEqual(
            self._encoder_layer_config(cfg).feed_forward_config.stack_config.num_layers,
            1,
        )

    def test_locked_preset_rejects_conflicting_overrides(self):
        presets = ExperimentPresets()

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                config.DATASET_OPTIONS[0],
                config_overrides={
                    "layer_norm_position": LayerNormPositionOptions.AFTER,
                },
            )

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                config.DATASET_OPTIONS[0],
                search_keys=["layer_norm_position"],
                search_mode=GridSearch(),
            )

        with self.assertRaisesRegex(ValueError, "PRE_NORM.*layer_norm_position"):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                config.DATASET_OPTIONS[0],
                GridSearch(),
                search_overrides={
                    "layer_norm_position": [LayerNormPositionOptions.AFTER],
                },
            )

    def test_presets_wire_config(self):
        presets = ExperimentPresets()

        cfg = presets.get_config(ExperimentPreset.BASELINE)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            TextLearnedPositionalEmbeddingConfig,
        )
        self.assertFalse(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentPreset.PRE_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )

        cfg = presets.get_config(ExperimentPreset.POST_NORM)[0]
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

        cfg = presets.get_config(ExperimentPreset.SINUSOIDAL)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            TextSinusoidalPositionalEmbeddingConfig,
        )

        cfg = presets.get_config(ExperimentPreset.CAUSAL)[0]
        self.assertTrue(self._encoder_layer_config(cfg).causal_attention_mask_flag)
        self.assertTrue(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentPreset.ATTENTION_BIAS)[0]
        attention_config = self._attention_config(cfg)
        self.assertTrue(attention_config.add_key_value_bias_flag)
        self.assertTrue(
            attention_config.projection_model_config.layer_config.layer_model_config.bias_flag
        )

    def test_model_uses_bert_pretraining_base_class_and_heads(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
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
            ExperimentPreset.BASELINE,
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
            ExperimentPreset.BASELINE,
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
