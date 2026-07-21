import importlib
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import torch
from torch import nn

import models.bert.linear.config as config
import models.bert.linear.dataset_options as dataset_options
import models.bert.linear.search_space as search_space
from emperor.attention import AttentionLayerState, SelfAttentionProjectionStrategy
from emperor.embedding.absolute import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConnectionOptions,
)
from emperor.transformer import (
    TransformerEncoderBlockLayer,
    TransformerEncoderLayer,
)
from model_runtime.packages import GridSearch, PresetLock, RandomSearch
from models.bert.linear._builder_adapter import linear_builder_kwargs_from_flat
from models.bert.linear.config_builder import (
    BertLinearConfigBuilder,
)
from models.bert.linear.model import Model
from models.bert.linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.bert.linear.runtime_defaults import runtime_from_flat
from models.bert.linear.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)
from models.config_overrides import iter_supported_config_keys, print_config_options
from models.parser import get_experiment_parser, resolve_experiment_mode
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)


def _default_builder_kwargs() -> dict:
    return linear_builder_kwargs_from_flat({}, config)


def _embedding_options() -> BertEmbeddingOptions:
    return _default_builder_kwargs()["embedding_options"]


def _encoder_options() -> TransformerEncoderOptions:
    return _default_builder_kwargs()["encoder_options"]


def _positional_embedding_options() -> TransformerPositionalEmbeddingOptions:
    return _default_builder_kwargs()["positional_embedding_options"]


def _attention_options() -> TransformerAttentionOptions:
    return _default_builder_kwargs()["attention_options"]


def _feed_forward_options() -> TransformerFeedForwardOptions:
    return _default_builder_kwargs()["feed_forward_options"]


def _mlm_head_options() -> BertMlmHeadOptions:
    return _default_builder_kwargs()["mlm_head_options"]


def _nsp_head_options() -> BertNspHeadOptions:
    return _default_builder_kwargs()["nsp_head_options"]


class TestBertLinearModel(unittest.TestCase):
    def test_runtime_defaults_describe_a_conventional_bert_block(self):
        self.assertEqual(
            config.STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(config.FF_NUM_LAYERS, 1)
        self.assertEqual(config.ATTN_STACK_ACTIVATION, ActivationOptions.DISABLED)
        self.assertFalse(config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG)
        self.assertEqual(
            config.FF_STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertFalse(config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG)

    def test_post_normalized_profile_has_no_extra_final_encoder_norm(self):
        model = Model(self._direct_config())

        self.assertFalse(
            any(name.startswith("encoder_layer_norm.") for name in model.state_dict())
        )

    def test_public_imports_remain_available(self):
        for module_name in (
            "models.bert.linear.config",
            "models.bert.linear.presets",
            "models.bert.linear.model",
            "models.bert.linear.config_builder",
            "models.bert.linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        self.assertEqual(
            Experiment()._public_model_id(),
            "bert/linear",
        )

    def test_option_group_build_matches_flat_kwargs(self):
        embedding_options = BertEmbeddingOptions(
            token_type_vocab_size=4,
            layer_norm_flag=False,
            dropout_probability=0.2,
        )
        encoder_options = TransformerEncoderOptions(
            hidden_dim=16,
            num_layers=1,
            activation=ActivationOptions.MISH,
            dropout_probability=0.1,
            layer_norm_position=LayerNormPositionOptions.AFTER,
            causal_attention_mask_flag=True,
        )
        positional_embedding_options = TransformerPositionalEmbeddingOptions(
            option=TextSinusoidalPositionalEmbeddingConfig,
            padding_idx=0,
            auto_expand_flag=True,
        )
        attention_options = TransformerAttentionOptions(
            num_heads=4,
            num_layers=2,
            bias_flag=True,
            add_key_value_bias_flag=True,
        )
        feed_forward_options = TransformerFeedForwardOptions(
            num_layers=2,
            bias_flag=False,
        )
        mlm_head_options = BertMlmHeadOptions(
            activation=ActivationOptions.SILU,
            dense_bias_flag=False,
            layer_norm_flag=False,
            decoder_bias_flag=False,
            decoder_weight_tying_flag=True,
        )
        nsp_head_options = BertNspHeadOptions(
            pooler_activation=ActivationOptions.SIGMOID,
            pooler_bias_flag=False,
            output_dim=3,
            head_bias_flag=False,
        )
        flat_kwargs = {
            "batch_size": 2,
            "learning_rate": 0.02,
            "input_dim": 32,
            "hidden_dim": encoder_options.hidden_dim,
            "output_dim": 32,
            "sequence_length": 8,
            "stack_num_layers": encoder_options.num_layers,
            "stack_activation": encoder_options.activation,
            "stack_dropout_probability": encoder_options.dropout_probability,
            "layer_norm_position": encoder_options.layer_norm_position,
            "causal_attention_mask_flag": (encoder_options.causal_attention_mask_flag),
            "positional_embedding_option": positional_embedding_options.option,
            "positional_embedding_padding_idx": (
                positional_embedding_options.padding_idx
            ),
            "positional_embedding_auto_expand_flag": (
                positional_embedding_options.auto_expand_flag
            ),
            "token_type_vocab_size": embedding_options.token_type_vocab_size,
            "embedding_layer_norm_flag": embedding_options.layer_norm_flag,
            "embedding_dropout_probability": (embedding_options.dropout_probability),
            "attn_num_heads": attention_options.num_heads,
            "attn_num_layers": attention_options.num_layers,
            "attn_bias_flag": attention_options.bias_flag,
            "attn_add_key_value_bias_flag": (attention_options.add_key_value_bias_flag),
            "ff_num_layers": feed_forward_options.num_layers,
            "ff_bias_flag": feed_forward_options.bias_flag,
            "mlm_activation": mlm_head_options.activation,
            "mlm_dense_bias_flag": mlm_head_options.dense_bias_flag,
            "mlm_layer_norm_flag": mlm_head_options.layer_norm_flag,
            "mlm_decoder_bias_flag": mlm_head_options.decoder_bias_flag,
            "mlm_decoder_weight_tying_flag": (
                mlm_head_options.decoder_weight_tying_flag
            ),
            "nsp_pooler_activation": nsp_head_options.pooler_activation,
            "nsp_pooler_bias_flag": nsp_head_options.pooler_bias_flag,
            "nsp_output_dim": nsp_head_options.output_dim,
            "nsp_head_bias_flag": nsp_head_options.head_bias_flag,
        }

        builder_kwargs = linear_builder_kwargs_from_flat(flat_kwargs, config)
        self.assertEqual(builder_kwargs["embedding_options"], embedding_options)
        self.assertEqual(builder_kwargs["mlm_head_options"], mlm_head_options)
        self.assertEqual(builder_kwargs["nsp_head_options"], nsp_head_options)
        self.assertNotIn("embedding_dropout_probability", builder_kwargs)

        flat_cfg = BertLinearConfigBuilder(**builder_kwargs).build()
        grouped_cfg = BertLinearConfigBuilder(
            batch_size=2,
            learning_rate=0.02,
            input_dim=32,
            output_dim=32,
            sequence_length=8,
            embedding_options=embedding_options,
            encoder_options=encoder_options,
            positional_embedding_options=positional_embedding_options,
            attention_options=attention_options,
            feed_forward_options=feed_forward_options,
            mlm_head_options=mlm_head_options,
            nsp_head_options=nsp_head_options,
        ).build()

        self.assertEqual(flat_cfg, grouped_cfg)
        self.assertEqual(
            grouped_cfg.experiment_config.boundary_config.embedding_options,
            embedding_options,
        )
        self.assertEqual(
            grouped_cfg.experiment_config.boundary_config.mlm_head_options,
            mlm_head_options,
        )
        self.assertEqual(
            grouped_cfg.experiment_config.boundary_config.nsp_head_options,
            nsp_head_options,
        )

    def test_direct_builder_rejects_legacy_flat_and_positional_args(self):
        for kwargs in (
            {"hidden_dim": 16},
            {"embedding_dropout_probability": 0.2},
            {"mlm_decoder_bias_flag": False},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    BertLinearConfigBuilder(**kwargs)

        with self.assertRaises(TypeError):
            BertLinearConfigBuilder(2)

    def test_typed_runtime_builds_the_flat_configuration(self):
        flat_options = {
            "batch_size": 2,
            "hidden_dim": 16,
            "stack_gate_flag": True,
        }
        runtime = runtime_from_flat(flat_options, config)

        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertEqual(
            BertLinearConfigBuilder(runtime=runtime).build(),
            BertLinearConfigBuilder(
                **linear_builder_kwargs_from_flat(flat_options, config)
            ).build(),
        )

    def test_bert_defaults_build_conventional_encoder_profile(self):
        self.assertEqual(config.STACK_NUM_LAYERS, 5)
        self.assertIs(
            config.STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.AFTER,
        )
        self.assertIs(config.LAYER_NORM_POSITION, config.STACK_LAYER_NORM_POSITION)
        self.assertTrue(config.ATTN_BIAS_FLAG)
        self.assertEqual(config.FF_STACK_HIDDEN_DIM, config.HIDDEN_DIM * 4)
        self.assertEqual(config.SUBMODULE_STACK_BIAS_FLAG, config.STACK_BIAS_FLAG)
        self.assertEqual(config.STACK_DROPOUT_PROBABILITY, 0.0)
        self.assertEqual(config.EMBEDDING_DROPOUT_PROBABILITY, 0.1)
        self.assertEqual(config.CALLBACK_EARLY_STOPPING_PATIENCE, 5)
        self.assertEqual(
            config.CALLBACK_EARLY_STOPPING_METRIC,
            "validation/loss",
        )

        cfg = BertLinearConfigBuilder().build()
        encoder_stack = self._encoder_stack_config(cfg)
        encoder_layer = self._encoder_layer_config(cfg)
        attention = self._attention_config(cfg)
        feed_forward_stack = self._feed_forward_stack_config(cfg)

        self.assertEqual(encoder_stack.num_layers, 5)
        self.assertEqual(
            encoder_layer.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertTrue(
            attention.projection_model_config.layer_config.layer_model_config.bias_flag
        )
        self.assertEqual(feed_forward_stack.hidden_dim, cfg.hidden_dim * 4)
        self.assertEqual(
            cfg.experiment_config.positional_embedding_config.padding_idx,
            0,
        )
        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options.dropout_probability,
            0.1,
        )
        self.assertEqual(config.HALTING_OUTPUT_DIM, 2)

    def test_main_stack_controls_build_into_encoder_stack(self):
        builder_kwargs = _default_builder_kwargs()
        builder_kwargs["stack_options"] = replace(
            builder_kwargs["stack_options"],
            residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            apply_output_pipeline_flag=False,
            bias_flag=False,
        )

        cfg = BertLinearConfigBuilder(**builder_kwargs).build()
        encoder_stack = self._encoder_stack_config(cfg)

        self.assertEqual(
            encoder_stack.layer_config.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(
            encoder_stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(encoder_stack.apply_output_pipeline_flag)

    def test_submodule_bias_inherits_main_stack_unless_explicitly_overridden(self):
        defaults = _default_builder_kwargs()
        stack_options = replace(defaults["stack_options"], bias_flag=False)
        layer_controller_options = replace(
            defaults["layer_controller_options"],
            stack_gate_flag=True,
        )

        inherited_cfg = BertLinearConfigBuilder(
            stack_options=stack_options,
            layer_controller_options=layer_controller_options,
        ).build()
        explicit_cfg = BertLinearConfigBuilder(
            stack_options=stack_options,
            submodule_stack_options=replace(
                defaults["submodule_stack_options"],
                bias_flag=True,
            ),
            layer_controller_options=layer_controller_options,
        ).build()
        flat_cfg = self._baseline_config(
            {
                "stack_bias_flag": False,
                "stack_gate_flag": True,
            }
        )

        self.assertFalse(self._encoder_gate_bias_flag(inherited_cfg))
        self.assertTrue(self._encoder_gate_bias_flag(explicit_cfg))
        self.assertFalse(self._encoder_gate_bias_flag(flat_cfg))

    def test_independent_encoder_controller_stacks_override_dimensions(self):
        builder_kwargs = _default_builder_kwargs()
        layer_controller_options = builder_kwargs["layer_controller_options"]
        dynamic_memory_options = builder_kwargs["dynamic_memory_options"]
        builder_kwargs.update(
            {
                "submodule_stack_options": replace(
                    builder_kwargs["submodule_stack_options"],
                    hidden_dim=17,
                ),
                "layer_controller_options": replace(
                    layer_controller_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_controller_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                        activation=ActivationOptions.TANH,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_controller_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "dynamic_memory_options": replace(
                    dynamic_memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        dynamic_memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        )

        cfg = BertLinearConfigBuilder(**builder_kwargs).build()
        stack = self._encoder_stack_config(cfg)
        gate_stack = stack.layer_config.gate_config.model_config
        halting_stack = stack.shared_halting_config.halting_gate_config
        memory_stack = stack.shared_memory_config.model_config

        self.assertEqual(stack.hidden_dim, cfg.hidden_dim)
        self.assertEqual((gate_stack.hidden_dim, gate_stack.num_layers), (23, 3))
        self.assertEqual(gate_stack.layer_config.activation, ActivationOptions.TANH)
        self.assertEqual(
            (halting_stack.hidden_dim, halting_stack.num_layers),
            (19, 2),
        )
        self.assertEqual(
            (memory_stack.hidden_dim, memory_stack.num_layers),
            (31, 4),
        )

    def test_attention_projection_stack_flat_overrides_are_applied(self):
        cfg = self._baseline_config(
            {
                "hidden_dim": 16,
                "attn_num_layers": 3,
                "attn_bias_flag": True,
                "attn_stack_hidden_dim": 24,
                "attn_stack_activation": ActivationOptions.MISH,
                "attn_stack_layer_norm_position": LayerNormPositionOptions.AFTER,
                "attn_stack_residual_connection_option": None,
                "attn_stack_dropout_probability": 0.2,
                "attn_stack_last_layer_bias_option": (LastLayerBiasOptions.DISABLED),
                "attn_stack_apply_output_pipeline_flag": False,
            }
        )

        stack = self._attention_projection_stack_config(cfg)

        self.assertIsInstance(stack, LayerStackConfig)
        self.assertEqual(stack.num_layers, 3)
        self.assertEqual(stack.hidden_dim, 24)
        self.assertFalse(stack.apply_output_pipeline_flag)
        self.assertEqual(
            stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertEqual(stack.layer_config.activation, ActivationOptions.MISH)
        self.assertEqual(
            stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(stack.layer_config.dropout_probability, 0.2)
        self.assertTrue(stack.layer_config.layer_model_config.bias_flag)

    def test_attention_projection_controls_attach_to_projection_stack(self):
        cfg = self._baseline_config(
            {
                "attn_gate_flag": True,
                "attn_gate_stack_independent_flag": True,
                "attn_gate_stack_hidden_dim": 12,
                "attn_gate_stack_num_layers": 3,
                "attn_halting_flag": True,
                "attn_memory_flag": True,
            }
        )

        stack = self._attention_projection_stack_config(cfg)

        self.assertIsNotNone(stack.layer_config.gate_config)
        self.assertIsNotNone(stack.layer_config.halting_config)
        self.assertIsNotNone(stack.shared_memory_config)
        self.assertEqual(
            stack.layer_config.halting_config.halting_gate_config.last_layer_bias_option,
            config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
        )
        gate_stack = stack.layer_config.gate_config.model_config
        self.assertEqual(gate_stack.hidden_dim, 12)
        self.assertEqual(gate_stack.num_layers, 3)
        self.assertEqual(stack.hidden_dim, cfg.hidden_dim)

    def test_independent_attention_controller_stacks_do_not_leak(self):
        builder_kwargs = _default_builder_kwargs()
        stack_options = builder_kwargs["attention_projection_stack_options"]
        layer_controller_options = builder_kwargs[
            "attention_projection_layer_controller_options"
        ]
        dynamic_memory_options = builder_kwargs[
            "attention_projection_dynamic_memory_options"
        ]
        builder_kwargs.update(
            {
                "attention_projection_stack_options": replace(
                    stack_options,
                    hidden_dim=17,
                ),
                "attention_projection_layer_controller_options": replace(
                    layer_controller_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_controller_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_controller_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "attention_projection_dynamic_memory_options": replace(
                    dynamic_memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        dynamic_memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        )

        cfg = BertLinearConfigBuilder(**builder_kwargs).build()
        stack = self._attention_projection_stack_config(cfg)
        gate_stack = stack.layer_config.gate_config.model_config
        halting_stack = stack.layer_config.halting_config.halting_gate_config
        memory_stack = stack.shared_memory_config.model_config
        encoder_stack = self._encoder_stack_config(cfg)
        feed_forward_stack = self._feed_forward_stack_config(cfg)

        self.assertEqual(stack.hidden_dim, 17)
        self.assertEqual((gate_stack.hidden_dim, gate_stack.num_layers), (23, 3))
        self.assertEqual(
            (halting_stack.hidden_dim, halting_stack.num_layers),
            (19, 2),
        )
        self.assertEqual(
            (memory_stack.hidden_dim, memory_stack.num_layers),
            (31, 4),
        )
        self.assertIsNone(encoder_stack.layer_config.gate_config)
        self.assertIsNone(encoder_stack.shared_memory_config)
        self.assertIsNone(feed_forward_stack.layer_config.gate_config)

    def test_attention_projection_recurrent_stack_forwards_batch(self):
        cfg = self._baseline_config(
            {
                "attn_recurrent_flag": True,
                "attn_recurrent_max_steps": 2,
            },
            batch_size=2,
        )

        attention_config = self._attention_config(cfg)
        projection_config = attention_config.projection_model_config
        self.assertIsInstance(projection_config, RecurrentLayerConfig)
        self.assertEqual(
            attention_config.projection_strategy,
            SelfAttentionProjectionStrategy.SEPARATE,
        )

        model = Model(cfg)
        batch = self._fake_bert_inputs(cfg, batch_size=2)
        mlm_logits, nsp_logits, auxiliary_loss = model(*batch)

        self.assertEqual(mlm_logits.shape, (2, cfg.sequence_length, cfg.output_dim))
        self.assertEqual(nsp_logits.shape, (2, 2))
        self.assertIsNotNone(auxiliary_loss)

    def test_feed_forward_stack_flat_overrides_are_applied(self):
        cfg = self._baseline_config(
            {
                "ff_num_layers": 3,
                "ff_bias_flag": False,
                "ff_stack_hidden_dim": 24,
                "ff_stack_activation": ActivationOptions.MISH,
                "ff_stack_layer_norm_position": LayerNormPositionOptions.AFTER,
                "ff_stack_residual_connection_option": None,
                "ff_stack_dropout_probability": 0.2,
                "ff_stack_last_layer_bias_option": LastLayerBiasOptions.DISABLED,
                "ff_stack_apply_output_pipeline_flag": False,
            }
        )

        stack = self._feed_forward_stack_config(cfg)

        self.assertIsInstance(stack, LayerStackConfig)
        self.assertEqual(stack.num_layers, 3)
        self.assertEqual(stack.hidden_dim, 24)
        self.assertFalse(stack.apply_output_pipeline_flag)
        self.assertEqual(
            stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertEqual(stack.layer_config.activation, ActivationOptions.MISH)
        self.assertEqual(
            stack.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(stack.layer_config.dropout_probability, 0.2)
        self.assertFalse(stack.layer_config.layer_model_config.bias_flag)

    def test_feed_forward_controls_attach_to_feed_forward_stack(self):
        cfg = self._baseline_config(
            {
                "ff_gate_flag": True,
                "ff_gate_stack_independent_flag": True,
                "ff_gate_stack_hidden_dim": 12,
                "ff_gate_stack_num_layers": 3,
                "ff_halting_flag": True,
                "ff_memory_flag": True,
            }
        )

        stack = self._feed_forward_stack_config(cfg)

        self.assertIsNotNone(stack.layer_config.gate_config)
        self.assertIsNotNone(stack.layer_config.halting_config)
        self.assertIsNotNone(stack.shared_memory_config)
        self.assertEqual(
            stack.layer_config.halting_config.halting_gate_config.last_layer_bias_option,
            config.FF_STACK_LAST_LAYER_BIAS_OPTION,
        )
        gate_stack = stack.layer_config.gate_config.model_config
        self.assertEqual(gate_stack.hidden_dim, 12)
        self.assertEqual(gate_stack.num_layers, 3)
        self.assertEqual(stack.hidden_dim, cfg.hidden_dim * 4)

    def test_independent_feed_forward_controller_stacks_do_not_leak(self):
        builder_kwargs = _default_builder_kwargs()
        stack_options = builder_kwargs["feed_forward_stack_options"]
        layer_controller_options = builder_kwargs[
            "feed_forward_layer_controller_options"
        ]
        dynamic_memory_options = builder_kwargs["feed_forward_dynamic_memory_options"]
        builder_kwargs.update(
            {
                "feed_forward_stack_options": replace(
                    stack_options,
                    hidden_dim=17,
                ),
                "feed_forward_layer_controller_options": replace(
                    layer_controller_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_controller_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_controller_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "feed_forward_dynamic_memory_options": replace(
                    dynamic_memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        dynamic_memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        )

        cfg = BertLinearConfigBuilder(**builder_kwargs).build()
        stack = self._feed_forward_stack_config(cfg)
        gate_stack = stack.layer_config.gate_config.model_config
        halting_stack = stack.layer_config.halting_config.halting_gate_config
        memory_stack = stack.shared_memory_config.model_config
        encoder_stack = self._encoder_stack_config(cfg)
        attention_stack = self._attention_projection_stack_config(cfg)

        self.assertEqual(stack.hidden_dim, 17)
        self.assertEqual((gate_stack.hidden_dim, gate_stack.num_layers), (23, 3))
        self.assertEqual(
            (halting_stack.hidden_dim, halting_stack.num_layers),
            (19, 2),
        )
        self.assertEqual(
            (memory_stack.hidden_dim, memory_stack.num_layers),
            (31, 4),
        )
        self.assertIsNone(encoder_stack.layer_config.gate_config)
        self.assertIsNone(encoder_stack.shared_memory_config)
        self.assertIsNone(attention_stack.layer_config.gate_config)

    def test_feed_forward_recurrent_stack_forwards_batch(self):
        cfg = self._baseline_config(
            {
                "ff_recurrent_flag": True,
                "ff_recurrent_max_steps": 2,
            },
            batch_size=2,
        )

        feed_forward_stack = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        self.assertIsInstance(feed_forward_stack, RecurrentLayerConfig)

        model = Model(cfg)
        batch = self._fake_bert_inputs(cfg, batch_size=2)
        mlm_logits, nsp_logits, auxiliary_loss = model(*batch)

        self.assertEqual(mlm_logits.shape, (2, cfg.sequence_length, cfg.output_dim))
        self.assertEqual(nsp_logits.shape, (2, 2))
        self.assertIsNotNone(auxiliary_loss)

    def test_recurrent_controller_stacks_follow_role_inheritance_rules(self):
        cases = {
            "encoder": {
                "stack_key": "submodule_stack_options",
                "layer_key": "layer_controller_options",
                "recurrent_key": "recurrent_controller_options",
                "config": self._encoder_config,
                "inherited_dimensions": (23, 19),
            },
            "attention": {
                "stack_key": "attention_projection_stack_options",
                "layer_key": "attention_projection_layer_controller_options",
                "recurrent_key": ("attention_projection_recurrent_controller_options"),
                "config": lambda cfg: (
                    self._attention_config(cfg).projection_model_config
                ),
                "inherited_dimensions": (17, 17),
            },
            "feed_forward": {
                "stack_key": "feed_forward_stack_options",
                "layer_key": "feed_forward_layer_controller_options",
                "recurrent_key": "feed_forward_recurrent_controller_options",
                "config": lambda cfg: (
                    self._encoder_layer_config(cfg).feed_forward_config.stack_config
                ),
                "inherited_dimensions": (17, 17),
            },
        }

        for role, case in cases.items():
            with self.subTest(role=role, source="inherited"):
                cfg = self._recurrent_controller_config(
                    stack_key=case["stack_key"],
                    layer_key=case["layer_key"],
                    recurrent_key=case["recurrent_key"],
                    independent_recurrent_flag=False,
                )
                recurrent_config = case["config"](cfg)
                self.assertIsInstance(recurrent_config, RecurrentLayerConfig)
                self.assertEqual(
                    self._recurrent_controller_dimensions(recurrent_config),
                    case["inherited_dimensions"],
                )

            with self.subTest(role=role, source="independent"):
                cfg = self._recurrent_controller_config(
                    stack_key=case["stack_key"],
                    layer_key=case["layer_key"],
                    recurrent_key=case["recurrent_key"],
                    independent_recurrent_flag=True,
                )
                recurrent_config = case["config"](cfg)
                self.assertEqual(
                    self._recurrent_controller_dimensions(recurrent_config),
                    (29, 31),
                )

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["models.bert.linear", "--preset", "baseline"]),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
        ):
            runpy.run_module(
                "models.bert.linear.__main__",
                run_name="__main__",
            )

        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args

        self.assertEqual(package.catalog_key, "bert/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(dict(plan.overrides), {})
        self.assertEqual(
            plan.datasets,
            tuple(
                dataset.__name__
                for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ]
            ),
        )
        self.assertEqual(execute_runs.call_args.kwargs["monitors"], ())

    def test_cli_and_config_listing_expose_flat_bert_boundary_flags(self):
        supported_keys = {key.lower() for key in iter_supported_config_keys(config)}
        self.assertTrue(
            {
                "token_type_vocab_size",
                "embedding_layer_norm_flag",
                "embedding_dropout_probability",
                "mlm_activation",
                "mlm_dense_bias_flag",
                "mlm_layer_norm_flag",
                "mlm_decoder_bias_flag",
                "mlm_decoder_weight_tying_flag",
                "nsp_pooler_activation",
                "nsp_pooler_bias_flag",
                "nsp_output_dim",
                "nsp_head_bias_flag",
            }
            <= supported_keys
        )

        parser = get_experiment_parser(
            ExperimentPreset.names(),
            "models.bert.linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--batch-size",
                "2",
                "--input-dim",
                "32",
                "--output-dim",
                "32",
                "--hidden-dim",
                "16",
                "--sequence-length",
                "8",
                "--stack-num-layers",
                "1",
                "--attn-num-heads",
                "4",
                "--token-type-vocab-size",
                "4",
                "--embedding-dropout-probability",
                "0.2",
                "--mlm-decoder-bias-flag",
                "false",
                "--nsp-output-dim",
                "3",
            ]
        )
        mode = resolve_experiment_mode(args, ExperimentPreset)
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides=mode.config_overrides,
        )[0]
        boundary = cfg.experiment_config.boundary_config

        self.assertEqual(boundary.embedding_options.token_type_vocab_size, 4)
        self.assertEqual(boundary.embedding_options.dropout_probability, 0.2)
        self.assertFalse(boundary.mlm_head_options.decoder_bias_flag)
        self.assertEqual(boundary.nsp_head_options.output_dim, 3)

        output = StringIO()
        with redirect_stdout(output):
            print_config_options("bert/linear")
        listing = output.getvalue()

        self.assertIn("--embedding-dropout-probability", listing)
        self.assertIn("--mlm-decoder-weight-tying-flag", listing)
        self.assertIn("--nsp-output-dim", listing)
        self.assertNotIn("--embedding-options", listing)
        self.assertNotIn("--mlm-head-options", listing)
        self.assertNotIn("--nsp-head-options", listing)

    def test_modern_preset_contract_is_exposed(self):
        presets = ExperimentPresets()

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
            ExperimentPreset.GATING: {
                "stack_gate_flag": True,
            },
            ExperimentPreset.HALTING: {
                "stack_halting_flag": True,
            },
            ExperimentPreset.GATING_HALTING: {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            },
            ExperimentPreset.MEMORY: {
                "memory_flag": True,
            },
            ExperimentPreset.GATING_MEMORY: {
                "stack_gate_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.HALTING_MEMORY: {
                "stack_halting_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.GATING_HALTING_MEMORY: {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT: {
                "recurrent_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING: {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
            },
            ExperimentPreset.RECURRENT_HALTING: {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
            },
            ExperimentPreset.RECURRENT_MEMORY: {
                "recurrent_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_HALTING: {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_HALTING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RESIDUAL: {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            },
            ExperimentPreset.RESIDUAL_POST_NORM: {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            },
            ExperimentPreset.RESIDUAL_GATING: {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "stack_gate_flag": True,
            },
            ExperimentPreset.RESIDUAL_HALTING: {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "stack_halting_flag": True,
            },
            ExperimentPreset.RESIDUAL_MEMORY: {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_RESIDUAL: {
                "recurrent_flag": True,
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            },
            ExperimentPreset.RECURRENT_POST_NORM: {
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            },
        }

        self.assertEqual(set(expected_overrides), set(ExperimentPreset))

        for preset, overrides in expected_overrides.items():
            self.assertEqual(presets.overrides_for_preset(preset), overrides)
            if not overrides:
                continue
            with self.subTest(preset=preset.name):
                self.assertEqual(
                    {
                        key: lock.value
                        for key, lock in presets.locks_for_preset(preset).items()
                    },
                    overrides,
                )

    def test_preset_numeric_ids_are_stable_and_append_residual_variants(self):
        expected_ids = {
            "BASELINE": 1,
            "PRE_NORM": 2,
            "POST_NORM": 3,
            "SINUSOIDAL": 4,
            "CAUSAL": 5,
            "ATTENTION_BIAS": 6,
            "GATING": 7,
            "HALTING": 8,
            "GATING_HALTING": 9,
            "MEMORY": 10,
            "GATING_MEMORY": 11,
            "HALTING_MEMORY": 12,
            "GATING_HALTING_MEMORY": 13,
            "RECURRENT": 14,
            "RECURRENT_GATING": 15,
            "RECURRENT_HALTING": 16,
            "RECURRENT_MEMORY": 17,
            "RECURRENT_GATING_HALTING": 18,
            "RECURRENT_GATING_MEMORY": 19,
            "RECURRENT_HALTING_MEMORY": 20,
            "RECURRENT_GATING_HALTING_MEMORY": 21,
            "RESIDUAL": 22,
            "RESIDUAL_POST_NORM": 23,
            "RESIDUAL_GATING": 24,
            "RESIDUAL_HALTING": 25,
            "RESIDUAL_MEMORY": 26,
            "RECURRENT_RESIDUAL": 27,
            "RECURRENT_POST_NORM": 28,
        }

        self.assertEqual(
            {preset.name: preset.value for preset in ExperimentPreset},
            expected_ids,
        )

    def test_controller_stack_config_constants_are_canonical_and_supported(self):
        canonical_names = {
            "ATTN_STACK_HIDDEN_DIM",
            "ATTN_STACK_LAYER_NORM_POSITION",
            "GATE_STACK_HIDDEN_DIM",
            "GATE_STACK_LAYER_NORM_POSITION",
            "GATE_STACK_BIAS_FLAG",
            "HALTING_STACK_HIDDEN_DIM",
            "HALTING_STACK_LAYER_NORM_POSITION",
            "HALTING_STACK_BIAS_FLAG",
            "FF_STACK_HIDDEN_DIM",
            "FF_STACK_LAYER_NORM_POSITION",
        }
        legacy_names = {name.replace("_STACK_", "_") for name in canonical_names}
        duplicate_stack_names = {
            "ATTN_STACK_NUM_LAYERS",
            "ATTN_STACK_BIAS_FLAG",
            "FF_STACK_NUM_LAYERS",
            "FF_STACK_BIAS_FLAG",
        }

        for name in canonical_names:
            with self.subTest(name=name):
                self.assertTrue(hasattr(config, name))
                self.assertNotIn(name, config.CONFIG_OVERRIDE_SKIP_KEYS)

        for name in legacy_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(config, name))
                self.assertNotIn(name, config.CONFIG_OVERRIDE_SKIP_KEYS)

        for name in duplicate_stack_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(config, name))
                self.assertNotIn(name, config.CONFIG_OVERRIDE_SKIP_KEYS)

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = ExperimentPresets()

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                expected_locks = presets.locks_for_preset(preset)
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

        for dataset in self._default_datasets():
            for preset in ExperimentPreset:
                with self.subTest(dataset=dataset.__name__, preset=preset.name):
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

        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
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

        for dataset in self._default_datasets():
            for preset in ExperimentPreset:
                with self.subTest(dataset=dataset.__name__, preset=preset.name):
                    cfg = presets.get_config(
                        preset,
                        dataset,
                        config_overrides=self._test_overrides(batch_size),
                    )[0]
                    model = Model(cfg)
                    datamodule = RandomBertPretrainingDataModule(
                        cfg,
                        batch_size=batch_size,
                        num_batches=1,
                    )

                    tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _test_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "hidden_dim": 16,
            "sequence_length": 8,
            "stack_num_layers": 2,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
        }

    def _baseline_config(self, overrides: dict, batch_size: int = 2):
        return ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides={
                "batch_size": batch_size,
                "input_dim": 32,
                "output_dim": 32,
                "hidden_dim": 16,
                "sequence_length": 8,
                "stack_num_layers": 1,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 4,
                **overrides,
            },
        )[0]

    def _attention_projection_stack_config(self, cfg):
        projection_config = self._attention_config(cfg).projection_model_config
        if isinstance(projection_config, RecurrentLayerConfig):
            return projection_config.block_config
        return projection_config

    def _feed_forward_stack_config(self, cfg):
        stack_config = self._encoder_layer_config(cfg).feed_forward_config.stack_config
        if isinstance(stack_config, RecurrentLayerConfig):
            return stack_config.block_config
        return stack_config

    def _direct_config(
        self,
        *,
        input_dim: int = 32,
        output_dim: int = 32,
        sequence_length: int = 8,
        embedding_options: BertEmbeddingOptions | None = None,
        mlm_head_options: BertMlmHeadOptions | None = None,
        nsp_head_options: BertNspHeadOptions | None = None,
    ):
        return BertLinearConfigBuilder(
            batch_size=2,
            input_dim=input_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            embedding_options=embedding_options,
            encoder_options=replace(
                _encoder_options(),
                hidden_dim=16,
                num_layers=1,
                dropout_probability=0.0,
            ),
            attention_options=replace(_attention_options(), num_heads=4),
            mlm_head_options=mlm_head_options,
            nsp_head_options=nsp_head_options,
        ).build()

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
            self._default_dataset(),
            RandomSearch(num_samples=2),
        )

        self.assertEqual(len(configs), 2)

    def test_search_keys_unknown_axis_raises(self):
        for search_key in ("bogus_axis", "encoder_options", "embedding_options"):
            with self.subTest(search_key=search_key):
                with self.assertRaises(ValueError) as ctx:
                    ExperimentPresets().get_config(
                        ExperimentPreset.BASELINE,
                        self._default_dataset(),
                        RandomSearch(num_samples=2),
                        search_keys=[search_key],
                    )

                self.assertIn("Unknown", str(ctx.exception))

    def test_search_space_matches_vit_shared_axes(self):
        self.assertEqual(search_space.SEARCH_SPACE_HIDDEN_DIM, [16, 32, 64, 128])
        self.assertEqual(search_space.SEARCH_SPACE_STACK_NUM_LAYERS, [1, 2, 4, 8])
        self.assertEqual(
            search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
            [
                LayerNormPositionOptions.BEFORE,
                LayerNormPositionOptions.AFTER,
            ],
        )
        self.assertIs(
            search_space.SEARCH_SPACE_STACK_LAYER_NORM_POSITION,
            search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
        )
        self.assertEqual(search_space.SEARCH_SPACE_ATTN_NUM_HEADS, [1, 2, 4])

    def test_search_applies_encoder_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            GridSearch(),
            search_keys=["hidden_dim", "stack_num_layers"],
        )

        self.assertEqual(
            len(configs),
            len(search_space.SEARCH_SPACE_HIDDEN_DIM)
            * len(search_space.SEARCH_SPACE_STACK_NUM_LAYERS),
        )
        self.assertEqual(
            {cfg.hidden_dim for cfg in configs},
            set(search_space.SEARCH_SPACE_HIDDEN_DIM),
        )
        self.assertEqual(
            {self._encoder_stack_config(cfg).num_layers for cfg in configs},
            set(search_space.SEARCH_SPACE_STACK_NUM_LAYERS),
        )

    def test_flat_search_keys_are_supported(self):
        cases = {
            "layer_norm_position": (
                search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
                lambda cfg: self._encoder_layer_config(cfg).layer_norm_position,
            ),
            "stack_layer_norm_position": (
                search_space.SEARCH_SPACE_STACK_LAYER_NORM_POSITION,
                lambda cfg: self._encoder_layer_config(cfg).layer_norm_position,
            ),
            "attn_num_heads": (
                search_space.SEARCH_SPACE_ATTN_NUM_HEADS,
                lambda cfg: self._attention_config(cfg).num_heads,
            ),
        }

        for search_key, (expected_values, accessor) in cases.items():
            with self.subTest(search_key=search_key):
                configs = ExperimentPresets().get_config(
                    ExperimentPreset.BASELINE,
                    self._default_dataset(),
                    GridSearch(),
                    search_keys=[search_key],
                )

                self.assertEqual(
                    [accessor(cfg) for cfg in configs],
                    expected_values,
                )

    def test_unlocked_overrides_update_flat_and_nested_config(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides={
                "hidden_dim": 24,
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

    def test_unlocked_grouped_overrides_update_boundary_and_encoder_config(self):
        embedding_options = replace(
            _embedding_options(),
            token_type_vocab_size=4,
            dropout_probability=0.2,
        )
        encoder_options = replace(
            _encoder_options(),
            hidden_dim=24,
            num_layers=2,
            dropout_probability=0.1,
        )
        mlm_head_options = replace(
            _mlm_head_options(),
            activation=ActivationOptions.MISH,
        )
        nsp_head_options = replace(
            _nsp_head_options(),
            output_dim=3,
        )
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides={
                "batch_size": 2,
                "input_dim": 32,
                "output_dim": 32,
                "sequence_length": 8,
                "embedding_options": embedding_options,
                "encoder_options": encoder_options,
                "attention_options": replace(_attention_options(), num_heads=4),
                "mlm_head_options": mlm_head_options,
                "nsp_head_options": nsp_head_options,
            },
        )[0]

        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(self._encoder_stack_config(cfg).num_layers, 2)
        self.assertEqual(self._encoder_layer_config(cfg).dropout_probability, 0.1)
        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options,
            embedding_options,
        )
        self.assertEqual(
            cfg.experiment_config.boundary_config.mlm_head_options,
            mlm_head_options,
        )
        self.assertEqual(
            cfg.experiment_config.boundary_config.nsp_head_options,
            nsp_head_options,
        )

    def test_flat_overrides_update_grouped_override_bases(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides={
                "input_dim": 32,
                "output_dim": 32,
                "encoder_options": replace(
                    _encoder_options(),
                    hidden_dim=16,
                    num_layers=1,
                    dropout_probability=0.2,
                ),
                "embedding_options": replace(
                    _embedding_options(),
                    dropout_probability=0.1,
                ),
                "hidden_dim": 24,
                "stack_num_layers": 2,
                "embedding_dropout_probability": 0.3,
            },
        )[0]

        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(self._encoder_stack_config(cfg).num_layers, 2)
        self.assertEqual(self._encoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options.dropout_probability,
            0.3,
        )

    def test_locked_preset_rejects_conflicting_overrides(self):
        presets = ExperimentPresets()

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ][0],
                config_overrides={
                    "layer_norm_position": LayerNormPositionOptions.AFTER,
                },
            )

        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ][0],
                search_keys=["layer_norm_position"],
                search_mode=GridSearch(),
            )

        with self.assertRaisesRegex(ValueError, "PRE_NORM.*layer_norm_position"):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                dataset_options.DATASET_OPTIONS_BY_TASK[
                    dataset_options.DEFAULT_EXPERIMENT_TASK
                ][0],
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
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
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
        self.assertTrue(self._attention_config(cfg).causal_attention_mask_flag)

        cfg = presets.get_config(ExperimentPreset.ATTENTION_BIAS)[0]
        attention_config = self._attention_config(cfg)
        self.assertTrue(attention_config.add_key_value_bias_flag)
        self.assertTrue(
            attention_config.projection_model_config.layer_config.layer_model_config.bias_flag
        )

        residual_cases = {
            ExperimentPreset.RESIDUAL: {},
            ExperimentPreset.RESIDUAL_POST_NORM: {"post_norm": True},
            ExperimentPreset.RESIDUAL_GATING: {"gate": True},
            ExperimentPreset.RESIDUAL_HALTING: {"halting": True},
            ExperimentPreset.RESIDUAL_MEMORY: {"memory": True},
            ExperimentPreset.RECURRENT_RESIDUAL: {"recurrent": True},
        }
        for preset, expected in residual_cases.items():
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(preset)[0]
                encoder_stack = self._encoder_stack_config(cfg)
                encoder_block = encoder_stack.layer_config

                self.assertEqual(
                    encoder_block.residual_config.option,
                    ResidualConnectionOptions.RESIDUAL,
                )
                self.assertEqual(
                    isinstance(self._encoder_config(cfg), RecurrentLayerConfig),
                    expected.get("recurrent", False),
                )
                self.assertEqual(
                    encoder_block.gate_config is not None,
                    expected.get("gate", False),
                )
                self.assertEqual(
                    encoder_stack.shared_halting_config is not None,
                    expected.get("halting", False),
                )
                self.assertEqual(
                    encoder_stack.shared_memory_config is not None,
                    expected.get("memory", False),
                )
                if expected.get("post_norm", False):
                    self.assertEqual(
                        self._encoder_layer_config(cfg).layer_norm_position,
                        LayerNormPositionOptions.AFTER,
                    )

        recurrent_post_norm = presets.get_config(ExperimentPreset.RECURRENT_POST_NORM)[
            0
        ]
        self.assertIsInstance(
            self._encoder_config(recurrent_post_norm),
            RecurrentLayerConfig,
        )
        self.assertEqual(
            self._encoder_layer_config(recurrent_post_norm).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )

    def test_model_uses_bert_pretraining_base_class_and_heads(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
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
        self.assertIsInstance(model.encoder_layer_norm, nn.Identity)
        self.assertIs(model.mlm_decoder.weight, model.token_embedding.weight)
        self.assertEqual(model.nsp_head.out_features, 2)
        state_dict = model.state_dict()
        boundary_state_keys = {
            key for key in state_dict if not key.startswith("transformer.")
        }
        self.assertEqual(
            boundary_state_keys,
            {
                "mlm_decoder_bias",
                "token_embedding.weight",
                "token_type_embedding.weight",
                "positional_embedding.embedding_model.weight",
                "embedding_layer_norm.weight",
                "embedding_layer_norm.bias",
                "mlm_dense.weight",
                "mlm_dense.bias",
                "mlm_layer_norm.weight",
                "mlm_layer_norm.bias",
                "mlm_decoder.weight",
                "pooler.weight",
                "pooler.bias",
                "nsp_head.weight",
                "nsp_head.bias",
            },
        )
        expected_shapes = {
            "mlm_decoder_bias": (cfg.output_dim,),
            "token_embedding.weight": (cfg.input_dim, cfg.hidden_dim),
            "token_type_embedding.weight": (2, cfg.hidden_dim),
            "positional_embedding.embedding_model.weight": (
                cfg.sequence_length + 1,
                cfg.hidden_dim,
            ),
            "embedding_layer_norm.weight": (cfg.hidden_dim,),
            "embedding_layer_norm.bias": (cfg.hidden_dim,),
            "mlm_dense.weight": (cfg.hidden_dim, cfg.hidden_dim),
            "mlm_dense.bias": (cfg.hidden_dim,),
            "mlm_layer_norm.weight": (cfg.hidden_dim,),
            "mlm_layer_norm.bias": (cfg.hidden_dim,),
            "mlm_decoder.weight": (cfg.output_dim, cfg.hidden_dim),
            "pooler.weight": (cfg.hidden_dim, cfg.hidden_dim),
            "pooler.bias": (cfg.hidden_dim,),
            "nsp_head.weight": (2, cfg.hidden_dim),
            "nsp_head.bias": (2,),
        }
        self.assertEqual(
            {key: tuple(state_dict[key].shape) for key in boundary_state_keys},
            expected_shapes,
        )

        reloaded_model = Model(cfg)
        incompatible_keys = reloaded_model.load_state_dict(state_dict, strict=True)
        self.assertEqual(incompatible_keys.missing_keys, [])
        self.assertEqual(incompatible_keys.unexpected_keys, [])
        self.assertIs(
            reloaded_model.mlm_decoder.weight,
            reloaded_model.token_embedding.weight,
        )

    def test_boundary_options_configure_embeddings_and_both_heads(self):
        embedding_options = BertEmbeddingOptions(
            token_type_vocab_size=4,
            layer_norm_flag=False,
            dropout_probability=0.25,
        )
        mlm_head_options = BertMlmHeadOptions(
            activation=ActivationOptions.MISH,
            dense_bias_flag=False,
            layer_norm_flag=False,
            decoder_bias_flag=False,
            decoder_weight_tying_flag=False,
        )
        nsp_head_options = BertNspHeadOptions(
            pooler_activation=ActivationOptions.SIGMOID,
            pooler_bias_flag=False,
            output_dim=3,
            head_bias_flag=False,
        )
        cfg = self._direct_config(
            embedding_options=embedding_options,
            mlm_head_options=mlm_head_options,
            nsp_head_options=nsp_head_options,
        )
        model = Model(cfg)

        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options,
            embedding_options,
        )
        self.assertEqual(
            cfg.experiment_config.boundary_config.mlm_head_options,
            mlm_head_options,
        )
        self.assertEqual(
            cfg.experiment_config.boundary_config.nsp_head_options,
            nsp_head_options,
        )
        self.assertEqual(model.token_type_embedding.num_embeddings, 4)
        self.assertIsInstance(model.embedding_layer_norm, nn.Identity)
        self.assertEqual(model.embedding_dropout.p, 0.25)
        self.assertIsNone(model.mlm_dense.bias)
        self.assertIsInstance(model.mlm_activation, nn.Mish)
        self.assertIsInstance(model.mlm_layer_norm, nn.Identity)
        self.assertIsNone(model.mlm_decoder_bias)
        self.assertIsNot(model.mlm_decoder.weight, model.token_embedding.weight)
        self.assertIsNone(model.pooler.bias)
        self.assertIsInstance(model.pooler_activation, nn.Sigmoid)
        self.assertEqual(model.nsp_head.out_features, 3)
        self.assertIsNone(model.nsp_head.bias)

        batch = self._fake_bert_inputs(cfg, batch_size=2)
        mlm_logits, nsp_logits, auxiliary_loss = model(*batch)

        self.assertEqual(mlm_logits.shape, (2, 8, 32))
        self.assertEqual(nsp_logits.shape, (2, 3))
        self.assertEqual(auxiliary_loss.shape, torch.Size([]))

    def test_mlm_decoder_weight_tying_can_be_enabled_or_disabled(self):
        tied_cfg = self._direct_config()
        untied_cfg = self._direct_config(
            mlm_head_options=replace(
                _mlm_head_options(),
                decoder_weight_tying_flag=False,
            )
        )

        tied_model = Model(tied_cfg)
        untied_model = Model(untied_cfg)

        self.assertIs(tied_model.mlm_decoder.weight, tied_model.token_embedding.weight)
        self.assertIsNot(
            untied_model.mlm_decoder.weight,
            untied_model.token_embedding.weight,
        )
        self.assertEqual(
            tied_model.mlm_decoder.weight.data_ptr(),
            tied_model.token_embedding.weight.data_ptr(),
        )
        self.assertNotEqual(
            untied_model.mlm_decoder.weight.data_ptr(),
            untied_model.token_embedding.weight.data_ptr(),
        )

    def test_untied_mlm_decoder_allows_mismatched_vocab_dimensions(self):
        cfg = self._direct_config(
            input_dim=29,
            output_dim=31,
            mlm_head_options=replace(
                _mlm_head_options(),
                decoder_weight_tying_flag=False,
            ),
        )
        model = Model(cfg)
        batch = self._fake_bert_inputs(cfg, batch_size=2)

        mlm_logits, nsp_logits, _auxiliary_loss = model(*batch)

        self.assertEqual(model.token_embedding.num_embeddings, 29)
        self.assertEqual(model.mlm_decoder.out_features, 31)
        self.assertEqual(mlm_logits.shape, (2, 8, 31))
        self.assertEqual(nsp_logits.shape, (2, 2))

    def test_tied_mlm_decoder_rejects_mismatched_vocab_dimensions(self):
        with self.assertRaisesRegex(ValueError, "weight tying.*input_dim.*output_dim"):
            self._direct_config(
                input_dim=29,
                output_dim=31,
                mlm_head_options=replace(
                    _mlm_head_options(),
                    decoder_weight_tying_flag=True,
                ),
            )

    def test_boundary_dimensions_must_be_positive(self):
        cases = {
            "input_dim": lambda: BertLinearConfigBuilder(input_dim=0).build(),
            "hidden_dim": lambda: BertLinearConfigBuilder(
                encoder_options=replace(_encoder_options(), hidden_dim=0)
            ).build(),
            "output_dim": lambda: BertLinearConfigBuilder(
                output_dim=0,
                mlm_head_options=replace(
                    _mlm_head_options(),
                    decoder_weight_tying_flag=False,
                ),
            ).build(),
            "sequence_length": lambda: BertLinearConfigBuilder(
                sequence_length=0
            ).build(),
            "token_type_vocab_size": lambda: BertLinearConfigBuilder(
                embedding_options=replace(
                    _embedding_options(),
                    token_type_vocab_size=0,
                )
            ).build(),
            "nsp_output_dim": lambda: BertLinearConfigBuilder(
                nsp_head_options=replace(_nsp_head_options(), output_dim=0)
            ).build(),
        }

        for field, build in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    build()

    def test_embedding_dropout_probability_must_be_bounded(self):
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "dropout_probability"):
                    BertLinearConfigBuilder(
                        embedding_options=replace(
                            _embedding_options(),
                            dropout_probability=probability,
                        )
                    ).build()

    def test_model_step_accepts_canonical_bert_pretraining_batch(self):
        batch_size = 2
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
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

    def test_forward_defaults_to_full_attention_and_zero_token_types(self):
        cfg = self._direct_config()
        model = Model(cfg).eval()
        input_ids = self._fake_bert_inputs(cfg, batch_size=2)[0]

        with torch.no_grad():
            default_outputs = model(input_ids)
            explicit_outputs = model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                token_type_ids=torch.zeros_like(input_ids),
            )

        for default, explicit in zip(default_outputs, explicit_outputs, strict=True):
            torch.testing.assert_close(default, explicit)

    def test_equal_batch_and_sequence_lengths_preserve_sample_isolation(self):
        cfg = self._direct_config(sequence_length=2)
        model = Model(cfg).eval()
        original_ids = torch.tensor([[2, 5], [7, 11]])
        changed_ids = torch.tensor([[2, 5], [13, 17]])
        attention_mask = torch.ones_like(original_ids)
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(
            changed_mlm[0],
            original_mlm[0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            changed_nsp[0],
            original_nsp[0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_mlm[1] - original_mlm[1])).item(),
            1e-6,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_nsp[1] - original_nsp[1])).item(),
            1e-6,
        )

    def test_default_encoder_uses_future_context(self):
        torch.manual_seed(17)
        model = Model(self._direct_config()).eval()
        original_ids = torch.tensor([[2, 3, 4, 5]])
        changed_ids = torch.tensor([[2, 3, 9, 5]])
        attention_mask = torch.ones_like(original_ids)
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, _, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, _, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        self.assertGreater(
            torch.max(torch.abs(changed_mlm[:, 0] - original_mlm[:, 0])).item(),
            1e-6,
        )

    def test_padding_token_values_do_not_change_visible_outputs(self):
        torch.manual_seed(19)
        model = Model(self._direct_config()).eval()
        original_ids = torch.tensor([[2, 3, 4, 5]])
        changed_ids = torch.tensor([[2, 3, 11, 12]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        token_type_ids = torch.zeros_like(original_ids)

        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                original_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed_ids,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(
            changed_mlm[:, :2],
            original_mlm[:, :2],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            changed_nsp,
            original_nsp,
            rtol=0.0,
            atol=0.0,
        )

    def test_encoder_auxiliary_loss_is_returned_and_added_to_pretraining_loss(self):
        batch_size = 2
        cfg = self._direct_config()
        model = Model(cfg)
        model.transformer = _AuxiliaryLossEncoder(0.25)
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(
            cfg,
            batch_size,
        )
        mlm_labels = torch.full_like(input_ids, -100)
        mlm_labels[:, 1] = input_ids[:, 1]
        next_sentence_labels = torch.tensor([0, 1])

        step_output = model._model_step_outputs(
            (
                input_ids,
                mlm_labels,
                attention_mask,
                token_type_ids,
                next_sentence_labels,
            )
        )

        torch.testing.assert_close(
            step_output.auxiliary_loss,
            step_output.total_loss.new_tensor(0.25),
        )
        torch.testing.assert_close(
            step_output.total_loss,
            step_output.mlm_loss
            + step_output.nsp_loss
            + step_output.total_loss.new_tensor(0.25),
        )

    def test_forward_converts_attention_mask_to_encoder_key_padding_mask(self):
        batch_size = 2
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
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
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
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

    def _encoder_stack_config(self, cfg):
        encoder_config = self._encoder_config(cfg)
        if isinstance(encoder_config, RecurrentLayerConfig):
            return encoder_config.block_config
        return encoder_config

    def _encoder_layer_config(self, cfg):
        return self._encoder_stack_config(cfg).layer_config.layer_model_config

    def _attention_config(self, cfg):
        return self._encoder_layer_config(cfg).attention_config

    def _encoder_gate_bias_flag(self, cfg) -> bool:
        gate_config = self._encoder_stack_config(cfg).layer_config.gate_config
        return gate_config.model_config.layer_config.layer_model_config.bias_flag

    def _recurrent_controller_config(
        self,
        *,
        stack_key: str,
        layer_key: str,
        recurrent_key: str,
        independent_recurrent_flag: bool,
    ):
        builder_kwargs = _default_builder_kwargs()
        layer_controller_options = builder_kwargs[layer_key]
        recurrent_controller_options = builder_kwargs[recurrent_key]
        recurrent_gate_stack_source = (
            recurrent_controller_options.recurrent_gate_stack_source
        )
        recurrent_halting_stack_source = (
            recurrent_controller_options.recurrent_halting_stack_source
        )
        if independent_recurrent_flag:
            recurrent_gate_stack_source = replace(
                recurrent_gate_stack_source,
                independent_flag=True,
                hidden_dim=29,
            )
            recurrent_halting_stack_source = replace(
                recurrent_halting_stack_source,
                independent_flag=True,
                hidden_dim=31,
            )

        builder_kwargs.update(
            {
                stack_key: replace(builder_kwargs[stack_key], hidden_dim=17),
                layer_key: replace(
                    layer_controller_options,
                    gate_stack_source=replace(
                        layer_controller_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                    ),
                    halting_stack_source=replace(
                        layer_controller_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                    ),
                ),
                recurrent_key: replace(
                    recurrent_controller_options,
                    recurrent_flag=True,
                    recurrent_gate_flag=True,
                    recurrent_gate_stack_source=recurrent_gate_stack_source,
                    recurrent_halting_flag=True,
                    recurrent_halting_stack_source=(recurrent_halting_stack_source),
                ),
            }
        )
        return BertLinearConfigBuilder(**builder_kwargs).build()

    def _recurrent_controller_dimensions(
        self,
        recurrent_config: RecurrentLayerConfig,
    ) -> tuple[int, int]:
        return (
            recurrent_config.gate_config.model_config.hidden_dim,
            recurrent_config.halting_config.halting_gate_config.hidden_dim,
        )

    def _default_datasets(self) -> list[type]:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _default_dataset(self) -> type:
        return self._default_datasets()[0]


class _AuxiliaryLossEncoder(nn.Module):
    def __init__(self, auxiliary_loss: float) -> None:
        super().__init__()
        self.auxiliary_loss = auxiliary_loss

    def forward(self, state: AttentionLayerState) -> AttentionLayerState:
        state.loss = state.hidden.new_tensor(self.auxiliary_loss)
        return state


if __name__ == "__main__":
    unittest.main()
