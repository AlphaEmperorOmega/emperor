import importlib
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
import torch.nn as nn

import models.vit.linear.config as config
import models.vit.linear.dataset_options as dataset_options
from emperor.attention import SelfAttentionProjectionStrategy
from emperor.embedding.absolute import (
    ImageLearnedPositionalEmbeddingConfig,
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.classifier import ClassifierExperiment
from emperor.halting import SoftHaltingConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)
from model_runtime.packages import PresetLock, iter_supported_config_keys
from models.catalog import model_package
from models.cli_selection import resolve_cli_selection
from models.config_overrides import print_config_options
from models.experiment_cli_parser import get_experiment_parser
from models.training_test_utils import (
    RandomImageClassificationDataModule,
    tiny_cpu_trainer,
)
from models.vit.linear.config_builder import VitLinearConfigBuilder
from models.vit.linear.model import Model
from models.vit.linear.presets import (
    Experiment,
    ExperimentPreset,
)
from models.vit.linear.runtime_defaults import runtime_from_flat
from models.vit.linear.runtime_options import RuntimeOptions

_TRANSFORMER_ENCODER_BLOCK_LAYER_TYPE = (
    TransformerEncoderBlockLayerConfig().registry_owner()
)
_TRANSFORMER_ENCODER_LAYER_TYPE = TransformerEncoderLayerConfig().registry_owner()


class TestVitLinearModel(unittest.TestCase):
    def test_public_imports_remain_available(self):
        for module_name in (
            "models.vit.linear.config",
            "models.vit.linear.presets",
            "models.vit.linear.model",
            "models.vit.linear.config_builder",
            "models.vit.linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

    def test_experiment_public_model_id_remains_catalog_id(self):
        experiment = Experiment(model_package=model_package("vit/linear"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "vit/linear",
        )

    def test_canonical_runtime_defaults_build_nested_config(self):
        runtime = model_package("vit/linear").bind_runtime_defaults(
            {
                "batch_size": 2,
                "learning_rate": 0.02,
                "input_dim": 192,
                "output_dim": 5,
                "image_patch_size": 4,
                "input_channels": 3,
                "image_height": 8,
                "patch_dropout_probability": 0.2,
                "patch_bias_flag": False,
                "hidden_dim": 16,
                "stack_num_layers": 2,
                "stack_activation": ActivationOptions.RELU,
                "stack_dropout_probability": 0.1,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
                "positional_embedding_option": (
                    ImageSinusoidalPositionalEmbeddingConfig
                ),
                "positional_embedding_padding_idx": None,
                "positional_embedding_auto_expand_flag": True,
                "attn_num_heads": 4,
                "attn_num_layers": 2,
                "attn_bias_flag": True,
                "attn_add_key_value_bias_flag": True,
                "ff_num_layers": 2,
                "ff_bias_flag": False,
            }
        )
        cfg = VitLinearConfigBuilder(runtime=runtime).build()

        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.learning_rate, 0.02)
        self.assertEqual(cfg.input_dim, 192)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(cfg.output_dim, 5)
        self.assertEqual(cfg.sequence_length, 5)
        self.assertEqual(
            cfg.experiment_config.patch_config.patch_size,
            4,
        )
        self.assertEqual(
            cfg.experiment_config.patch_config.dropout_probability,
            0.2,
        )
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            ImageSinusoidalPositionalEmbeddingConfig,
        )
        self.assertTrue(
            self._attention_config(cfg).add_key_value_bias_flag,
        )
        self.assertEqual(
            self._feed_forward_stack_config(cfg).num_layers,
            2,
        )

    def test_direct_builder_rejects_legacy_flat_and_positional_args(self):
        with self.assertRaises(TypeError):
            VitLinearConfigBuilder(hidden_dim=16)

        with self.assertRaises(TypeError):
            VitLinearConfigBuilder(2)

    def test_typed_runtime_builds_the_model_config(self):
        runtime = runtime_from_flat(
            {
                "batch_size": 2,
                "learning_rate": 0.02,
                "hidden_dim": 16,
                "stack_num_layers": 1,
            },
            config,
        )

        self.assertIsInstance(runtime, RuntimeOptions)
        cfg = VitLinearConfigBuilder(runtime=runtime).build()

        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.learning_rate, 0.02)
        self.assertEqual(cfg.hidden_dim, 16)
        self.assertEqual(self._encoder_stack_config(cfg).num_layers, 1)

    def test_output_config_factory_builds_cls_classification_head(self):
        overrides = self._small_image_overrides(batch_size=2)
        overrides["output_dim"] = 7
        overrides["stack_activation"] = ActivationOptions.RELU
        overrides["stack_dropout_probability"] = 0.2
        cfg = self._build_config(**overrides)

        output_cfg = cfg.experiment_config.output_config

        self.assertIsInstance(output_cfg, LayerConfig)
        self.assertEqual(output_cfg.input_dim, cfg.hidden_dim)
        self.assertEqual(output_cfg.output_dim, cfg.output_dim)
        self.assertIsInstance(output_cfg.layer_model_config, LinearLayerConfig)
        self.assertTrue(output_cfg.layer_model_config.bias_flag)

    def test_encoder_stack_fields_build_through_runtime_defaults(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
            stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            stack_apply_output_pipeline_flag=False,
            stack_bias_flag=False,
        )
        encoder_cfg = self._encoder_stack_config(cfg)

        self.assertEqual(
            encoder_cfg.layer_config.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(
            encoder_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(encoder_cfg.apply_output_pipeline_flag)
        self.assertTrue(
            cfg.experiment_config.output_config.layer_model_config.bias_flag
        )

    def test_stack_gate_flag_creates_encoder_block_gate_config(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            stack_gate_flag=True,
        )

        self.assertIsNotNone(self._encoder_block_config(cfg).gate_config)

    def test_stack_halting_flag_creates_shared_encoder_halting_config(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            stack_halting_flag=True,
        )

        encoder_stack = self._encoder_stack_config(cfg)
        self.assertIsNone(self._encoder_block_config(cfg).halting_config)
        self.assertIsNotNone(encoder_stack.shared_halting_config)

    def test_soft_halting_option_is_forwarded_but_rejected_until_supported(self):
        overrides = self._small_image_overrides(batch_size=2)
        overrides["stack_num_layers"] = 2
        cfg = self._build_config(
            **overrides,
            stack_halting_flag=True,
            halting_option=SoftHaltingConfig,
        )

        encoder_stack = self._encoder_stack_config(cfg)
        self.assertIsInstance(
            encoder_stack.shared_halting_config,
            SoftHaltingConfig,
        )
        with self.assertRaisesRegex(ValueError, "does not implement"):
            encoder_stack.build()

    def test_memory_flag_creates_encoder_stack_shared_memory(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            memory_flag=True,
        )

        self.assertIsNotNone(self._encoder_stack_config(cfg).shared_memory_config)

    def test_recurrent_flag_wraps_encoder_stack_and_forwards_batch(self):
        batch_size = 2
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=batch_size),
            recurrent_flag=True,
            recurrent_max_steps=2,
        )
        model = Model(cfg)
        images = torch.randn(batch_size, 3, 8, 8)

        logits = model(images)

        self.assertIsInstance(
            cfg.experiment_config.encoder_config,
            RecurrentLayerConfig,
        )
        self.assertEqual(logits.shape, (batch_size, 5))

    def test_independent_gate_stack_overrides_controller_dimensions(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            stack_gate_flag=True,
            gate_stack_independent_flag=True,
            gate_stack_hidden_dim=29,
            gate_stack_num_layers=3,
            gate_stack_activation=ActivationOptions.SILU,
        )
        gate_cfg = self._encoder_block_config(cfg).gate_config.model_config

        self.assertEqual(gate_cfg.hidden_dim, 29)
        self.assertEqual(gate_cfg.num_layers, 3)
        self.assertEqual(gate_cfg.layer_config.activation, ActivationOptions.SILU)

    def test_default_encoder_stack_preserves_transformer_layer_behavior(self):
        cfg = VitLinearConfigBuilder().build()
        encoder_cfg = cfg.experiment_config.encoder_config
        outer_layer_cfg = encoder_cfg.layer_config
        inner_layer_cfg = outer_layer_cfg.layer_model_config
        feed_forward_stack_cfg = inner_layer_cfg.feed_forward_config.stack_config

        self.assertNotIsInstance(encoder_cfg, RecurrentLayerConfig)
        self.assertIsNone(outer_layer_cfg.gate_config)
        self.assertIsNone(outer_layer_cfg.halting_config)
        self.assertIsNone(encoder_cfg.shared_memory_config)
        self.assertIsNone(outer_layer_cfg.residual_config)
        self.assertEqual(
            outer_layer_cfg.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertEqual(outer_layer_cfg.dropout_probability, 0.0)
        self.assertEqual(
            inner_layer_cfg.layer_norm_position,
            config.LAYER_NORM_POSITION,
        )
        self.assertEqual(
            inner_layer_cfg.dropout_probability,
            config.STACK_DROPOUT_PROBABILITY,
        )
        self.assertEqual(
            inner_layer_cfg.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        attention_projection_layer_cfg = (
            inner_layer_cfg.attention_config.projection_model_config.layer_config
        )
        self.assertEqual(
            attention_projection_layer_cfg.layer_model_config.bias_flag,
            config.ATTN_BIAS_FLAG,
        )
        attention_projection_stack_cfg = (
            inner_layer_cfg.attention_config.projection_model_config
        )
        self.assertEqual(attention_projection_stack_cfg.hidden_dim, config.HIDDEN_DIM)
        self.assertEqual(
            attention_projection_stack_cfg.num_layers,
            config.ATTN_NUM_LAYERS,
        )
        self.assertEqual(
            attention_projection_stack_cfg.layer_config.activation,
            config.STACK_ACTIVATION,
        )
        self.assertEqual(
            attention_projection_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertIsNone(attention_projection_stack_cfg.layer_config.residual_config)
        self.assertEqual(
            attention_projection_stack_cfg.layer_config.dropout_probability,
            0.0,
        )
        self.assertEqual(
            attention_projection_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DEFAULT,
        )
        self.assertTrue(attention_projection_stack_cfg.apply_output_pipeline_flag)
        feed_forward_layer_cfg = (
            inner_layer_cfg.feed_forward_config.stack_config.layer_config
        )
        self.assertTrue(feed_forward_layer_cfg.layer_model_config.bias_flag)
        self.assertIsNone(feed_forward_stack_cfg.layer_config.gate_config)
        self.assertIsNone(feed_forward_stack_cfg.layer_config.halting_config)
        self.assertIsNone(feed_forward_stack_cfg.shared_memory_config)
        self.assertEqual(feed_forward_stack_cfg.hidden_dim, config.FF_STACK_HIDDEN_DIM)
        self.assertEqual(feed_forward_stack_cfg.num_layers, config.FF_NUM_LAYERS)
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.activation,
            config.FF_STACK_ACTIVATION,
        )
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertIsNone(feed_forward_stack_cfg.layer_config.residual_config)
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.dropout_probability,
            config.STACK_DROPOUT_PROBABILITY,
        )
        self.assertEqual(
            feed_forward_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DEFAULT,
        )
        self.assertTrue(feed_forward_stack_cfg.apply_output_pipeline_flag)

    def test_attention_projection_stack_overrides_configure_projection_stack(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            attn_stack_hidden_dim=29,
            attn_stack_activation=ActivationOptions.SILU,
            attn_stack_residual_connection_option=(ResidualConnectionOptions.RESIDUAL),
            attn_stack_dropout_probability=0.2,
            attn_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            attn_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            attn_stack_apply_output_pipeline_flag=False,
        )
        projection_stack_cfg = self._attention_projection_stack_config(cfg)

        self.assertEqual(projection_stack_cfg.hidden_dim, 29)
        self.assertEqual(
            projection_stack_cfg.layer_config.activation,
            ActivationOptions.SILU,
        )
        self.assertEqual(
            projection_stack_cfg.layer_config.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(projection_stack_cfg.layer_config.dropout_probability, 0.2)
        self.assertEqual(
            projection_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(
            projection_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(projection_stack_cfg.apply_output_pipeline_flag)

    def test_attention_num_layers_and_bias_remain_canonical_for_projection_stack(self):
        overrides = self._small_image_overrides(batch_size=2)
        overrides["attn_num_layers"] = 3
        overrides["attn_bias_flag"] = True
        cfg = self._build_config(
            **overrides,
        )
        projection_stack_cfg = self._attention_projection_stack_config(cfg)

        self.assertEqual(projection_stack_cfg.num_layers, 3)
        self.assertTrue(projection_stack_cfg.layer_config.layer_model_config.bias_flag)

    def test_attention_projection_controls_attach_to_projection_stack_only(self):
        overrides = self._small_image_overrides(batch_size=2)
        overrides["attn_num_layers"] = 2
        cfg = self._build_config(
            **overrides,
            attn_stack_gate_flag=True,
            attn_stack_halting_flag=True,
            attn_memory_flag=True,
        )
        projection_stack_cfg = self._attention_projection_stack_config(cfg)

        self.assertIsNotNone(projection_stack_cfg.layer_config.gate_config)
        self.assertIsNotNone(projection_stack_cfg.layer_config.halting_config)
        self.assertIsNotNone(projection_stack_cfg.shared_memory_config)
        self.assertIsNone(self._encoder_block_config(cfg).gate_config)
        self.assertIsNone(self._encoder_stack_config(cfg).shared_memory_config)
        self.assertIsNone(self._feed_forward_stack_config(cfg).layer_config.gate_config)

    def test_attention_projection_recurrent_flag_wraps_stack_and_forwards_batch(self):
        batch_size = 2
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=batch_size),
            attn_recurrent_flag=True,
            attn_recurrent_max_steps=2,
        )
        model = Model(cfg)
        images = torch.randn(batch_size, 3, 8, 8)

        logits = model(images)

        self.assertIsInstance(
            self._attention_config(cfg).projection_model_config,
            RecurrentLayerConfig,
        )
        self.assertEqual(
            self._attention_config(cfg).projection_strategy,
            SelfAttentionProjectionStrategy.SEPARATE,
        )
        self.assertEqual(logits.shape, (batch_size, 5))

    def test_independent_attention_projection_stacks_override_only_controllers(self):
        overrides = self._small_image_overrides(batch_size=2)
        overrides["attn_num_layers"] = 2
        cfg = self._build_config(
            **overrides,
            attn_stack_hidden_dim=17,
            attn_stack_gate_flag=True,
            attn_gate_stack_independent_flag=True,
            attn_gate_stack_hidden_dim=23,
            attn_gate_stack_num_layers=3,
            attn_gate_stack_activation=ActivationOptions.TANH,
            attn_stack_halting_flag=True,
            attn_halting_stack_independent_flag=True,
            attn_halting_stack_hidden_dim=19,
            attn_halting_stack_num_layers=2,
            attn_memory_flag=True,
            attn_memory_stack_independent_flag=True,
            attn_memory_stack_hidden_dim=31,
            attn_memory_stack_num_layers=4,
        )
        projection_stack_cfg = self._attention_projection_stack_config(cfg)
        gate_stack_cfg = projection_stack_cfg.layer_config.gate_config.model_config
        halting_stack_cfg = (
            projection_stack_cfg.layer_config.halting_config.halting_gate_config
        )
        memory_stack_cfg = projection_stack_cfg.shared_memory_config.model_config

        self.assertEqual(projection_stack_cfg.hidden_dim, 17)
        self.assertEqual(gate_stack_cfg.hidden_dim, 23)
        self.assertEqual(gate_stack_cfg.num_layers, 3)
        self.assertEqual(gate_stack_cfg.layer_config.activation, ActivationOptions.TANH)
        self.assertEqual(halting_stack_cfg.hidden_dim, 19)
        self.assertEqual(halting_stack_cfg.num_layers, 2)
        self.assertEqual(memory_stack_cfg.hidden_dim, 31)
        self.assertEqual(memory_stack_cfg.num_layers, 4)

    def test_attention_recurrent_controller_stacks_inherit_attention_stack(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            attn_stack_hidden_dim=17,
            attn_gate_stack_independent_flag=True,
            attn_gate_stack_hidden_dim=23,
            attn_halting_stack_independent_flag=True,
            attn_halting_stack_hidden_dim=19,
            attn_recurrent_flag=True,
            attn_recurrent_stack_gate_flag=True,
            attn_recurrent_stack_halting_flag=True,
        )
        recurrent_cfg = self._attention_config(cfg).projection_model_config
        recurrent_gate_stack_cfg = recurrent_cfg.gate_config.model_config
        recurrent_halting_stack_cfg = recurrent_cfg.halting_config.halting_gate_config

        self.assertEqual(recurrent_gate_stack_cfg.hidden_dim, 17)
        self.assertEqual(recurrent_halting_stack_cfg.hidden_dim, 17)

    def test_feed_forward_stack_overrides_configure_transformer_ff_stack(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_stack_hidden_dim=29,
            ff_stack_activation=ActivationOptions.SILU,
            ff_stack_residual_connection_option=(ResidualConnectionOptions.RESIDUAL),
            ff_stack_dropout_probability=0.2,
            ff_stack_layer_norm_position=LayerNormPositionOptions.AFTER,
            ff_stack_last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            ff_stack_apply_output_pipeline_flag=False,
        )
        feed_forward_stack_cfg = self._feed_forward_stack_config(cfg)

        self.assertEqual(feed_forward_stack_cfg.hidden_dim, 29)
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.activation,
            ActivationOptions.SILU,
        )
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(feed_forward_stack_cfg.layer_config.dropout_probability, 0.2)
        self.assertEqual(
            feed_forward_stack_cfg.layer_config.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(
            feed_forward_stack_cfg.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(feed_forward_stack_cfg.apply_output_pipeline_flag)

    def test_feed_forward_num_layers_and_bias_remain_canonical(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_num_layers=3,
            ff_bias_flag=False,
        )
        feed_forward_stack_cfg = self._feed_forward_stack_config(cfg)

        self.assertEqual(feed_forward_stack_cfg.num_layers, 3)
        self.assertFalse(
            feed_forward_stack_cfg.layer_config.layer_model_config.bias_flag
        )

    def test_feed_forward_gate_flag_attaches_gate_to_ff_stack_only(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_stack_gate_flag=True,
        )

        self.assertIsNotNone(
            self._feed_forward_stack_config(cfg).layer_config.gate_config
        )
        self.assertIsNone(self._encoder_block_config(cfg).gate_config)

    def test_feed_forward_halting_flag_attaches_halting_to_ff_stack_only(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_stack_halting_flag=True,
        )

        self.assertIsNotNone(
            self._feed_forward_stack_config(cfg).layer_config.halting_config
        )
        self.assertIsNone(self._encoder_block_config(cfg).halting_config)

    def test_feed_forward_memory_flag_attaches_shared_memory_to_ff_stack_only(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_memory_flag=True,
        )

        self.assertIsNotNone(self._feed_forward_stack_config(cfg).shared_memory_config)
        self.assertIsNone(self._encoder_stack_config(cfg).shared_memory_config)

    def test_feed_forward_recurrent_flag_wraps_ff_stack_and_forwards_batch(self):
        batch_size = 2
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=batch_size),
            ff_recurrent_flag=True,
            ff_recurrent_max_steps=2,
        )
        model = Model(cfg)
        images = torch.randn(batch_size, 3, 8, 8)

        logits = model(images)

        self.assertIsInstance(
            self._feed_forward_config(cfg).stack_config,
            RecurrentLayerConfig,
        )
        self.assertEqual(logits.shape, (batch_size, 5))

    def test_independent_feed_forward_controller_stacks_override_only_controllers(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_stack_hidden_dim=17,
            ff_stack_gate_flag=True,
            ff_gate_stack_independent_flag=True,
            ff_gate_stack_hidden_dim=23,
            ff_gate_stack_num_layers=3,
            ff_gate_stack_activation=ActivationOptions.TANH,
            ff_stack_halting_flag=True,
            ff_halting_stack_independent_flag=True,
            ff_halting_stack_hidden_dim=19,
            ff_halting_stack_num_layers=2,
            ff_memory_flag=True,
            ff_memory_stack_independent_flag=True,
            ff_memory_stack_hidden_dim=31,
            ff_memory_stack_num_layers=4,
        )
        feed_forward_stack_cfg = self._feed_forward_stack_config(cfg)
        gate_stack_cfg = feed_forward_stack_cfg.layer_config.gate_config.model_config
        halting_stack_cfg = (
            feed_forward_stack_cfg.layer_config.halting_config.halting_gate_config
        )
        memory_stack_cfg = feed_forward_stack_cfg.shared_memory_config.model_config

        self.assertEqual(feed_forward_stack_cfg.hidden_dim, 17)
        self.assertEqual(gate_stack_cfg.hidden_dim, 23)
        self.assertEqual(gate_stack_cfg.num_layers, 3)
        self.assertEqual(gate_stack_cfg.layer_config.activation, ActivationOptions.TANH)
        self.assertEqual(halting_stack_cfg.hidden_dim, 19)
        self.assertEqual(halting_stack_cfg.num_layers, 2)
        self.assertEqual(memory_stack_cfg.hidden_dim, 31)
        self.assertEqual(memory_stack_cfg.num_layers, 4)

    def test_feed_forward_recurrent_controller_stacks_inherit_ff_stack(self):
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=2),
            ff_stack_hidden_dim=17,
            ff_gate_stack_independent_flag=True,
            ff_gate_stack_hidden_dim=23,
            ff_halting_stack_independent_flag=True,
            ff_halting_stack_hidden_dim=19,
            ff_recurrent_flag=True,
            ff_recurrent_stack_gate_flag=True,
            ff_recurrent_stack_halting_flag=True,
        )
        recurrent_cfg = self._feed_forward_config(cfg).stack_config
        recurrent_gate_stack_cfg = recurrent_cfg.gate_config.model_config
        recurrent_halting_stack_cfg = recurrent_cfg.halting_config.halting_gate_config

        self.assertEqual(recurrent_gate_stack_cfg.hidden_dim, 17)
        self.assertEqual(recurrent_halting_stack_cfg.hidden_dim, 17)

    def test_module_entrypoint_resolves_cli_without_training(self):
        with (
            patch.object(sys, "argv", ["linear", "--preset", "baseline"]),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
            self.assertRaises(SystemExit) as exit_context,
        ):
            runpy.run_module(
                "models.vit.linear.__main__",
                run_name="__main__",
            )

        self.assertEqual(exit_context.exception.code, 0)
        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args

        self.assertEqual(package.catalog_key, "vit/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertIsNone(plan.search)
        self.assertEqual(dict(plan.overrides), {})
        self.assertEqual(
            plan.datasets,
            (self._default_dataset().__name__,),
        )

    def test_cli_and_config_listing_expose_flat_vit_flags(self):
        supported_keys = {key.lower() for key in iter_supported_config_keys(config)}
        self.assertTrue(
            {
                "batch_size",
                "learning_rate",
                "input_dim",
                "output_dim",
                "num_epochs",
                "hidden_dim",
                "image_patch_size",
                "attn_num_heads",
                "layer_norm_position",
                "stack_num_layers",
            }
            <= supported_keys
        )
        self.assertNotIn("sequence_length", supported_keys)

        package = model_package("vit/linear")
        parser = get_experiment_parser(package)
        parser.parse_args(["--preset", "baseline", "--batch-size", "2"])
        parser.parse_args(["--preset", "baseline", "--hidden-dim", "16"])
        parser.parse_args(["--preset", "baseline", "--image-patch-size", "4"])
        parser.parse_args(["--preset", "baseline", "--attn-num-heads", "4"])

        output = StringIO()
        with redirect_stdout(output):
            print_config_options("vit/linear")
        listing = output.getvalue()

        self.assertIn("--batch-size", listing)
        self.assertIn("--hidden-dim", listing)
        self.assertIn("--image-patch-size", listing)
        self.assertIn("--attn-num-heads", listing)
        self.assertIn("--stack-num-layers", listing)
        self.assertIn("--layer-norm-position", listing)
        self.assertNotIn("encoder_options", listing)
        self.assertNotIn("--sequence-length", listing)
        self.assertNotIn("--patch-options", listing)
        self.assertNotIn("--encoder-options", listing)

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
            ExperimentPreset.GATING: {
                "stack_gate_flag": True,
            },
            ExperimentPreset.HALTING: {
                "stack_halting_flag": True,
            },
            ExperimentPreset.MEMORY: {
                "memory_flag": True,
            },
            ExperimentPreset.GATING_HALTING: {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
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
            ExperimentPreset.RECURRENT: {
                "recurrent_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING: {
                "recurrent_flag": True,
                "recurrent_stack_gate_flag": True,
            },
            ExperimentPreset.RECURRENT_HALTING: {
                "recurrent_flag": True,
                "recurrent_stack_halting_flag": True,
            },
            ExperimentPreset.RECURRENT_MEMORY: {
                "recurrent_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_HALTING: {
                "recurrent_flag": True,
                "recurrent_stack_gate_flag": True,
                "recurrent_stack_halting_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_stack_gate_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_HALTING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_stack_halting_flag": True,
                "memory_flag": True,
            },
            ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: {
                "recurrent_flag": True,
                "recurrent_stack_gate_flag": True,
                "recurrent_stack_halting_flag": True,
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

        presets = model_package("vit/linear").presets

        self.assertEqual(
            {
                preset: presets.overrides_for_preset(preset)
                for preset in ExperimentPreset
            },
            expected_overrides,
        )
        for preset, overrides in expected_overrides.items():
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

    def test_preset_locks_are_exposed_with_reasons(self):
        presets = model_package("vit/linear").presets

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
        presets = model_package("vit/linear").presets
        dataset = self._default_dataset()

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    dataset,
                    config_overrides=self._runtime_overrides(batch_size),
                )[0]
                model = Model(cfg)
                images = self._fake_batch(dataset, batch_size)

                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_baseline_forwards_all_image_datasets(self):
        batch_size = 2
        presets = model_package("vit/linear").presets

        for dataset in self._default_datasets():
            with self.subTest(dataset=dataset.__name__):
                cfg = presets.get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides=self._runtime_overrides(batch_size),
                )[0]
                model = Model(cfg)
                images = self._fake_batch(dataset, batch_size)

                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

                self.assertEqual(logits.shape, (batch_size, dataset.num_classes))

    def test_equal_batch_and_sequence_lengths_preserve_sample_isolation(self):
        batch_size = 5
        cfg = self._build_config(
            **self._small_image_overrides(batch_size=batch_size),
        )
        model = Model(cfg).eval()
        torch.manual_seed(149)
        original_images = torch.randn(batch_size, 3, 8, 8)
        changed_images = original_images.clone()
        changed_images[1] = torch.randn_like(changed_images[1]).mul_(4.0)

        with torch.no_grad():
            original_logits = model(original_images)
            changed_logits = model(changed_images)

        torch.testing.assert_close(
            changed_logits[0],
            original_logits[0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            torch.max(torch.abs(changed_logits[1] - original_logits[1])).item(),
            1e-6,
        )

    def test_all_presets_train_one_epoch(self):
        batch_size = 2
        presets = model_package("vit/linear").presets
        dataset = self._default_dataset()

        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = presets.get_config(
                    preset,
                    dataset,
                    config_overrides=self._runtime_overrides(batch_size),
                )[0]
                model = Model(cfg)
                datamodule = RandomImageClassificationDataModule(
                    dataset,
                    batch_size=batch_size,
                )

                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def test_canonical_flat_overrides_update_top_level_and_nested_config(self):
        dataset = self._default_dataset()
        cfg = model_package("vit/linear").presets.get_config(
            ExperimentPreset.BASELINE,
            dataset,
            config_overrides={
                "batch_size": 2,
                "hidden_dim": 24,
                "stack_num_layers": 2,
                "stack_activation": ActivationOptions.RELU,
                "stack_dropout_probability": 0.2,
                "patch_dropout_probability": 0.3,
                "attn_num_heads": 4,
            },
        )[0]

        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(cfg.experiment_config.encoder_config.num_layers, 2)
        self.assertEqual(self._encoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(self._attention_config(cfg).dropout_probability, 0.2)
        self.assertEqual(self._attention_projection_stack_config(cfg).hidden_dim, 24)
        self.assertEqual(cfg.experiment_config.patch_config.dropout_probability, 0.3)
        self.assertIsInstance(
            cfg.experiment_config.output_config,
            LayerConfig,
        )
        self.assertEqual(cfg.experiment_config.output_config.input_dim, 24)
        self.assertIsInstance(
            cfg.experiment_config.output_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_flat_cli_overrides_update_grouped_builder_options(self):
        dataset = self._default_dataset()
        package = model_package("vit/linear")
        parser = get_experiment_parser(package)
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--batch-size",
                "2",
                "--hidden-dim",
                "24",
                "--stack-num-layers",
                "2",
                "--image-patch-size",
                "4",
                "--attn-num-heads",
                "4",
            ]
        )
        mode = resolve_cli_selection(args, package, ExperimentPreset)

        cfg = model_package("vit/linear").presets.get_config(
            ExperimentPreset.BASELINE,
            dataset,
            config_overrides=mode.config_overrides,
        )[0]

        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(cfg.experiment_config.encoder_config.num_layers, 2)
        self.assertEqual(cfg.experiment_config.patch_config.patch_size, 4)
        self.assertEqual(
            cfg.sequence_length,
            self._expected_sequence_length(dataset.default_height, 4),
        )
        self.assertEqual(self._attention_config(cfg).num_heads, 4)

    def test_canonical_layer_norm_flag_updates_encoder_options(self):
        package = model_package("vit/linear")
        parser = get_experiment_parser(package)
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--batch-size",
                "2",
                "--layer-norm-position",
                "AFTER",
            ]
        )
        mode = resolve_cli_selection(args, package, ExperimentPreset)
        cfg = package.presets.get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides=mode.config_overrides,
        )[0]

        self.assertIs(
            mode.config_overrides["layer_norm_position"],
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--preset",
                    "baseline",
                    "--stack-layer-norm-position",
                    "AFTER",
                ]
            )

    def test_grouped_override_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "encoder_options"):
            model_package("vit/linear").presets.get_config(
                ExperimentPreset.BASELINE,
                self._default_dataset(),
                config_overrides={"encoder_options": object()},
            )

    def test_locked_preset_rejects_conflicting_overrides(self):
        presets = model_package("vit/linear").presets

        with self.assertRaisesRegex(ValueError, "POST_NORM.*layer_norm_position"):
            presets.get_config(
                ExperimentPreset.POST_NORM,
                self._default_dataset(),
                config_overrides={
                    "batch_size": 2,
                    "layer_norm_position": LayerNormPositionOptions.BEFORE,
                },
            )

    def test_model_inherits_classifier_experiment(self):
        cfg = model_package("vit/linear").presets.get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
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
        dataset = self._default_dataset()
        cfg = model_package("vit/linear").presets.get_config(
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
        cfg = model_package("vit/linear").presets.get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        embedding = model.positional_embedding.embedding_model

        self.assertIsNone(embedding.padding_idx)
        self.assertTrue(embedding.weight.requires_grad)

    def test_encoder_is_built_from_transformer_encoder_block_layers(self):
        cfg = model_package("vit/linear").presets.get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides=self._test_overrides(batch_size=2),
        )[0]
        model = Model(cfg)

        encoder_layers = self._encoder_layers(model)

        self.assertGreater(len(encoder_layers), 0)
        for layer in encoder_layers:
            self.assertIsInstance(layer, _TRANSFORMER_ENCODER_BLOCK_LAYER_TYPE)
            self.assertIsInstance(layer.model, _TRANSFORMER_ENCODER_LAYER_TYPE)

    def test_model_step_accepts_classifier_batch(self):
        batch_size = 2
        dataset = self._default_dataset()
        cfg = model_package("vit/linear").presets.get_config(
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
        dataset = self._default_dataset()
        cfg = model_package("vit/linear").presets.get_config(
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
        presets = model_package("vit/linear").presets

        for dataset in self._default_datasets():
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
                self.assertEqual(
                    positional_cfg.num_embeddings, expected_sequence_length - 1
                )
                self.assertTrue(positional_cfg.class_token_flag)
                self.assertEqual(
                    attention_cfg.target_sequence_length, expected_sequence_length
                )
                self.assertEqual(
                    attention_cfg.source_sequence_length, expected_sequence_length
                )

    def test_presets_wire_config_variants(self):
        presets = model_package("vit/linear").presets

        cfg = presets.get_config(ExperimentPreset.BASELINE)[0]
        self.assertIsInstance(
            cfg.experiment_config.positional_embedding_config,
            ImageLearnedPositionalEmbeddingConfig,
        )
        self.assertEqual(
            self._encoder_layer_config(cfg).layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
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

        cfg = presets.get_config(ExperimentPreset.GATING)[0]
        self.assertIsNotNone(self._encoder_block_config(cfg).gate_config)

        cfg = presets.get_config(ExperimentPreset.HALTING)[0]
        self.assertIsNotNone(self._encoder_stack_config(cfg).shared_halting_config)

        cfg = presets.get_config(ExperimentPreset.MEMORY)[0]
        self.assertIsNotNone(self._encoder_stack_config(cfg).shared_memory_config)

        cfg = presets.get_config(ExperimentPreset.RESIDUAL)[0]
        self.assertEqual(
            self._encoder_block_config(cfg).residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )

        cfg = presets.get_config(ExperimentPreset.RECURRENT)[0]
        self.assertIsInstance(self._encoder_config(cfg), RecurrentLayerConfig)

    def test_invalid_patch_size_for_image_height_raises(self):
        with self.assertRaises(ValueError):
            self._build_config(image_height=30, image_patch_size=4)

    def _default_datasets(self) -> list[type]:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _default_dataset(self) -> type:
        return self._default_datasets()[0]

    def _runtime_overrides(self, batch_size: int) -> dict:
        return {"batch_size": batch_size}

    def _test_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 4,
        }

    def _small_image_overrides(self, batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "output_dim": 5,
            "image_patch_size": 4,
            "image_height": 8,
            "input_channels": 3,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 4,
        }

    def _build_config(self, **runtime_defaults):
        runtime = model_package("vit/linear").bind_runtime_defaults(runtime_defaults)
        return VitLinearConfigBuilder(runtime=runtime).build()

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

    def _encoder_stack_config(self, cfg):
        encoder_cfg = self._encoder_config(cfg)
        if isinstance(encoder_cfg, RecurrentLayerConfig):
            return encoder_cfg.block_config
        return encoder_cfg

    def _encoder_block_config(self, cfg):
        return self._encoder_stack_config(cfg).layer_config

    def _encoder_layer_config(self, cfg):
        return self._encoder_block_config(cfg).layer_model_config

    def _feed_forward_config(self, cfg):
        return self._encoder_layer_config(cfg).feed_forward_config

    def _feed_forward_stack_config(self, cfg):
        stack_config = self._feed_forward_config(cfg).stack_config
        if isinstance(stack_config, RecurrentLayerConfig):
            return stack_config.block_config
        return stack_config

    def _attention_config(self, cfg):
        return self._encoder_layer_config(cfg).attention_config

    def _attention_projection_stack_config(self, cfg):
        projection_config = self._attention_config(cfg).projection_model_config
        if isinstance(projection_config, RecurrentLayerConfig):
            return projection_config.block_config
        return projection_config


class _AuxiliaryLossEncoder(nn.Module):
    def __init__(self, auxiliary_loss: float):
        super().__init__()
        self.auxiliary_loss = auxiliary_loss

    def forward(self, state):
        state.loss = state.hidden.new_tensor(self.auxiliary_loss)
        return state


if __name__ == "__main__":
    unittest.main()
