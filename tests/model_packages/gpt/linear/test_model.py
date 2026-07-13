import importlib
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import models.gpt.linear.config as config
import models.gpt.linear.dataset_options as dataset_options
import models.gpt.linear.search_space as search_space
import torch
import torch.nn as nn
from emperor.attention import SelfAttentionProjectionStrategy
from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.embedding.absolute.core.config import (
    TextLearnedPositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.transformer import (
    TransformerDecoderBlockLayer,
    TransformerDecoderLayer,
    TransformerDecoderLayerState,
)
from models.catalog import MODEL_CATALOG, catalog_entry
from models.config_overrides import iter_supported_config_keys, print_config_options
from models.gpt.linear import (
    Experiment,
    ExperimentConfig,
    ExperimentPreset,
    ExperimentPresets,
    GptLinearConfigBuilder,
    GptLmHeadOptions,
    Model,
)
from models.gpt.linear._builder_adapter import linear_builder_kwargs_from_flat
from models.gpt.linear.runtime_defaults import runtime_from_flat
from models.gpt.linear.runtime_options import (
    GptEmbeddingOptions,
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)
from models.parser import get_experiment_parser, resolve_experiment_mode
from models.training_test_utils import (
    RandomLanguageModelDataModule,
    tiny_cpu_trainer,
)

from model_runtime.packages import GridSearch, PresetLock, RandomSearch


class TestGptLinearModel(unittest.TestCase):
    backend_module_name = "LinearLayer"

    def config(self, **overrides):
        return GptLinearConfigBuilder(
            input_dim=16,
            output_dim=16,
            sequence_length=6,
            **overrides,
        ).build()

    def test_public_imports_and_catalog_identity(self):
        self.assertTrue(issubclass(Model, LanguageModelExperiment))
        self.assertIsNotNone(Experiment)
        self.assertIsNotNone(ExperimentConfig)
        self.assertIsNotNone(ExperimentPresets)
        self.assertEqual(
            MODEL_CATALOG["gpt/linear"].module_path,
            "models.gpt.linear",
        )

    def test_presets_are_contiguous_buildable_and_always_causal(self):
        presets = ExperimentPresets()
        self.assertNotIn("CAUSAL", ExperimentPreset.__members__)
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, len(ExperimentPreset) + 1)),
        )
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                configs = presets.get_config(preset)
                self.assertTrue(configs)
                decoder_config = configs[0].experiment_config.decoder_config
                block_config = getattr(decoder_config, "block_config", decoder_config)
                layer_config = block_config.layer_config.layer_model_config
                self.assertTrue(layer_config.causal_attention_mask_flag)
                self.assertTrue(
                    layer_config.self_attention_config.causal_attention_mask_flag
                )
                self.assertIsNone(layer_config.cross_attention_config)

    def test_forward_shape_tied_head_and_backend_construction(self):
        model = Model(self.config()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        logits, auxiliary_loss = model(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 4, 16))
        self.assertEqual(tuple(auxiliary_loss.shape), ())
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNone(model.lm_head.bias)
        self.assertIn(
            self.backend_module_name,
            {type(module).__name__ for module in model.modules()},
        )
        decoder_layers = [
            module
            for module in model.modules()
            if isinstance(module, TransformerDecoderLayer)
        ]
        self.assertTrue(decoder_layers)
        self.assertTrue(
            all(layer.cross_attention_model is None for layer in decoder_layers)
        )

    def test_lm_head_options_allow_untied_biased_projection(self):
        cfg = self.config(
            lm_head_options=GptLmHeadOptions(
                weight_tying_flag=False,
                bias_flag=True,
            )
        )
        model = Model(cfg)
        self.assertIsNot(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNotNone(model.lm_head.bias)

    def test_tied_head_rejects_unequal_vocabularies(self):
        with self.assertRaises(ValueError):
            GptLinearConfigBuilder(
                input_dim=15,
                output_dim=16,
                sequence_length=4,
            ).build()

    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(7)
        model = Model(self.config()).eval()
        original = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 9, 10]])

        with torch.no_grad():
            original_logits, _ = model(original)
            changed_logits, _ = model(changed_future)

        torch.testing.assert_close(
            original_logits[:, :3],
            changed_logits[:, :3],
            rtol=0.0,
            atol=0.0,
        )

    def test_training_loss_has_gradient_flow(self):
        torch.manual_seed(11)
        model = Model(self.config())
        input_ids = torch.tensor([[1, 2, 3], [3, 4, 5]])
        labels = torch.tensor([[2, 3, 4], [4, 5, 6]])
        loss = model._model_step((input_ids, labels))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(any(torch.any(gradient.abs() > 0) for gradient in gradients))

    def test_generate_is_greedy_deterministic_and_restores_mode(self):
        torch.manual_seed(13)
        model = Model(self.config())
        prompt = torch.tensor([[1, 2, 3]])
        model.eval()
        with torch.no_grad():
            prompt_logits, _ = model(prompt)
        expected_next = prompt_logits[:, -1].argmax(dim=-1)

        model.train()
        first = model.generate(prompt, max_new_tokens=1)
        second = model.generate(prompt, max_new_tokens=1)

        self.assertTrue(model.training)
        self.assertEqual(tuple(first.shape), (1, 4))
        torch.testing.assert_close(first[:, :3], prompt)
        torch.testing.assert_close(first[:, -1], expected_next)
        torch.testing.assert_close(first, second)

    def test_generate_rejects_invalid_requests(self):
        model = Model(self.config())
        invalid_cases = (
            (torch.tensor([1, 2]), 1, (ValueError, TypeError)),
            (torch.empty((1, 0), dtype=torch.long), 1, (ValueError,)),
            (torch.tensor([[1, -1]]), 1, (ValueError,)),
            (torch.tensor([[1, 16]]), 1, (ValueError,)),
            (torch.tensor([[1, 2]]), -1, (ValueError,)),
            (torch.tensor([[1, 2]]), 5, (ValueError,)),
        )
        for input_ids, max_new_tokens, errors in invalid_cases:
            with self.subTest(
                shape=tuple(input_ids.shape),
                max_new_tokens=max_new_tokens,
            ):
                with self.assertRaises(errors):
                    model.generate(input_ids, max_new_tokens)

    def test_all_public_modules_import_and_catalog_resolves(self):
        for module_name in (
            "models.gpt.linear.config",
            "models.gpt.linear.presets",
            "models.gpt.linear.model",
            "models.gpt.linear.config_builder",
            "models.gpt.linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                self.assertEqual(
                    importlib.import_module(module_name).__name__,
                    module_name,
                )
        self.assertEqual(Experiment()._public_model_id(), "gpt/linear")
        self.assertIsNotNone(catalog_entry("gpt/linear"))

    def test_package_avoids_bert_and_gpt_sibling_construction_imports(self):
        package_dir = Path(config.__file__).resolve().parent
        blocked = (
            "models.bert.",
            "models.gpt.linear_adaptive.",
            "models.gpt.expert_linear",
        )
        for path in package_dir.glob("*.py"):
            if path.name == "test_model.py":
                continue
            source = path.read_text(encoding="utf-8")
            for import_path in blocked:
                with self.subTest(path=path.name, import_path=import_path):
                    self.assertNotIn(import_path, source)

    def test_grouped_options_match_flat_configuration(self):
        embedding_options = GptEmbeddingOptions(
            layer_norm_flag=False,
            dropout_probability=0.2,
        )
        decoder_options = TransformerDecoderOptions(
            hidden_dim=16,
            num_layers=2,
            activation=ActivationOptions.MISH,
            dropout_probability=0.1,
            layer_norm_position=LayerNormPositionOptions.AFTER,
        )
        positional_options = TransformerPositionalEmbeddingOptions(
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
        lm_head_options = GptLmHeadOptions(
            weight_tying_flag=True,
            bias_flag=False,
        )
        flat_options = {
            "batch_size": 2,
            "learning_rate": 0.02,
            "input_dim": 32,
            "output_dim": 32,
            "sequence_length": 8,
            "embedding_layer_norm_flag": False,
            "embedding_dropout_probability": 0.2,
            "hidden_dim": 16,
            "stack_num_layers": 2,
            "stack_activation": ActivationOptions.MISH,
            "stack_dropout_probability": 0.1,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
            "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
            "positional_embedding_padding_idx": 0,
            "positional_embedding_auto_expand_flag": True,
            "attn_num_heads": 4,
            "attn_num_layers": 2,
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
            "ff_num_layers": 2,
            "ff_bias_flag": False,
            "lm_head_weight_tying_flag": True,
            "lm_head_bias_flag": False,
        }
        adapted = linear_builder_kwargs_from_flat(flat_options, config)
        self.assertEqual(adapted["embedding_options"], embedding_options)
        self.assertEqual(adapted["decoder_options"], decoder_options)
        self.assertEqual(adapted["positional_embedding_options"], positional_options)
        self.assertEqual(adapted["attention_options"], attention_options)
        self.assertEqual(adapted["feed_forward_options"], feed_forward_options)
        self.assertEqual(adapted["lm_head_options"], lm_head_options)
        flat_cfg = GptLinearConfigBuilder(**adapted).build()
        grouped_cfg = GptLinearConfigBuilder(
            batch_size=2,
            learning_rate=0.02,
            input_dim=32,
            output_dim=32,
            sequence_length=8,
            embedding_options=embedding_options,
            decoder_options=decoder_options,
            positional_embedding_options=positional_options,
            attention_options=attention_options,
            feed_forward_options=feed_forward_options,
            lm_head_options=lm_head_options,
        ).build()
        self.assertEqual(flat_cfg, grouped_cfg)

    def test_typed_runtime_builds_flat_configuration(self):
        flat_options = {
            "batch_size": 2,
            "input_dim": 32,
            "output_dim": 32,
            "hidden_dim": 16,
            "stack_gate_flag": True,
        }
        runtime = runtime_from_flat(flat_options, config)
        self.assertIsInstance(runtime, RuntimeOptions)
        self.assertEqual(
            GptLinearConfigBuilder(runtime=runtime).build(),
            GptLinearConfigBuilder(
                **linear_builder_kwargs_from_flat(flat_options, config)
            ).build(),
        )

    def test_builder_rejects_positional_unknown_and_removed_options(self):
        with self.assertRaises(TypeError):
            GptLinearConfigBuilder(2)
        for kwargs in (
            {"unknown_option": 7},
            {"causal_attention_mask_flag": False},
            {"token_type_vocab_size": 4},
            {"mlm_decoder_bias_flag": False},
            {"nsp_output_dim": 3},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    GptLinearConfigBuilder(**kwargs)

    def test_shared_defaults_build_permanently_causal_decoder(self):
        self.assertEqual(config.STACK_NUM_LAYERS, 5)
        self.assertIs(
            config.STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertEqual(config.EMBEDDING_DROPOUT_PROBABILITY, 0.1)
        self.assertTrue(config.LM_HEAD_WEIGHT_TYING_FLAG)
        self.assertFalse(config.LM_HEAD_BIAS_FLAG)
        cfg = GptLinearConfigBuilder().build()
        stack = self._decoder_stack_config(cfg)
        layer = self._decoder_layer_config(cfg)
        attention = self._attention_config(cfg)
        self.assertEqual(stack.num_layers, 5)
        self.assertEqual(
            layer.layer_norm_position,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertTrue(layer.causal_attention_mask_flag)
        self.assertTrue(attention.causal_attention_mask_flag)
        self.assertIsNone(layer.cross_attention_config)
        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options.dropout_probability,
            0.1,
        )

    def test_main_stack_controls_build_into_decoder_stack(self):
        defaults = self._default_builder_kwargs()
        cfg = GptLinearConfigBuilder(
            **{
                **defaults,
                "stack_options": replace(
                    defaults["stack_options"],
                    residual_connection_option=(ResidualConnectionOptions.RESIDUAL),
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    bias_flag=False,
                ),
            }
        ).build()
        stack = self._decoder_stack_config(cfg)
        self.assertEqual(
            stack.layer_config.residual_connection_option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(
            stack.last_layer_bias_option,
            LastLayerBiasOptions.DISABLED,
        )
        self.assertFalse(stack.apply_output_pipeline_flag)

    def test_submodule_bias_inherits_main_stack_unless_overridden(self):
        defaults = self._default_builder_kwargs()
        stack_options = replace(defaults["stack_options"], bias_flag=False)
        layer_options = replace(
            defaults["layer_controller_options"],
            stack_gate_flag=True,
        )
        inherited = GptLinearConfigBuilder(
            stack_options=stack_options,
            layer_controller_options=layer_options,
        ).build()
        explicit = GptLinearConfigBuilder(
            stack_options=stack_options,
            submodule_stack_options=replace(
                defaults["submodule_stack_options"],
                bias_flag=True,
            ),
            layer_controller_options=layer_options,
        ).build()
        self.assertFalse(self._decoder_gate_bias_flag(inherited))
        self.assertTrue(self._decoder_gate_bias_flag(explicit))

    def test_independent_decoder_controller_stacks_override_dimensions(self):
        defaults = self._default_builder_kwargs()
        layer_options = defaults["layer_controller_options"]
        memory_options = defaults["dynamic_memory_options"]
        cfg = GptLinearConfigBuilder(
            **{
                **defaults,
                "submodule_stack_options": replace(
                    defaults["submodule_stack_options"],
                    hidden_dim=17,
                ),
                "layer_controller_options": replace(
                    layer_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "dynamic_memory_options": replace(
                    memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        ).build()
        stack = self._decoder_stack_config(cfg)
        gate = stack.layer_config.gate_config.model_config
        halting = stack.layer_config.halting_config.halting_gate_config
        memory = stack.shared_memory_config.model_config
        self.assertEqual((gate.hidden_dim, gate.num_layers), (23, 3))
        self.assertEqual((halting.hidden_dim, halting.num_layers), (19, 2))
        self.assertEqual((memory.hidden_dim, memory.num_layers), (31, 4))

    def test_attention_projection_stack_flat_overrides_are_applied(self):
        cfg = self._baseline_config(
            {
                "attn_num_layers": 3,
                "attn_bias_flag": True,
                "attn_stack_hidden_dim": 24,
                "attn_stack_activation": ActivationOptions.MISH,
                "attn_stack_layer_norm_position": LayerNormPositionOptions.AFTER,
                "attn_stack_dropout_probability": 0.2,
                "attn_stack_last_layer_bias_option": (LastLayerBiasOptions.DISABLED),
                "attn_stack_apply_output_pipeline_flag": False,
            }
        )
        stack = self._attention_projection_stack_config(cfg)
        self.assertIsInstance(stack, LayerStackConfig)
        self.assertEqual((stack.num_layers, stack.hidden_dim), (3, 24))
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

    def test_attention_controls_attach_only_to_projection_stack(self):
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
        gate = stack.layer_config.gate_config.model_config
        self.assertEqual((gate.hidden_dim, gate.num_layers), (12, 3))
        self.assertIsNone(self._decoder_stack_config(cfg).layer_config.gate_config)
        self.assertIsNone(self._feed_forward_stack_config(cfg).layer_config.gate_config)

    def test_independent_attention_controller_stacks_do_not_leak(self):
        defaults = self._default_builder_kwargs()
        stack_options = defaults["attention_projection_stack_options"]
        layer_options = defaults["attention_projection_layer_controller_options"]
        memory_options = defaults["attention_projection_dynamic_memory_options"]
        cfg = GptLinearConfigBuilder(
            **{
                **defaults,
                "attention_projection_stack_options": replace(
                    stack_options,
                    hidden_dim=17,
                ),
                "attention_projection_layer_controller_options": replace(
                    layer_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "attention_projection_dynamic_memory_options": replace(
                    memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        ).build()
        stack = self._attention_projection_stack_config(cfg)
        gate = stack.layer_config.gate_config.model_config
        halting = stack.layer_config.halting_config.halting_gate_config
        memory = stack.shared_memory_config.model_config
        self.assertEqual(stack.hidden_dim, 17)
        self.assertEqual((gate.hidden_dim, gate.num_layers), (23, 3))
        self.assertEqual((halting.hidden_dim, halting.num_layers), (19, 2))
        self.assertEqual((memory.hidden_dim, memory.num_layers), (31, 4))
        self.assertIsNone(self._decoder_stack_config(cfg).layer_config.gate_config)
        self.assertIsNone(self._feed_forward_stack_config(cfg).layer_config.gate_config)

    def test_attention_projection_recurrence_forwards_batch(self):
        cfg = self._baseline_config(
            {
                "attn_recurrent_flag": True,
                "attn_recurrent_max_steps": 2,
            }
        )
        attention = self._attention_config(cfg)
        self.assertIsInstance(
            attention.projection_model_config,
            RecurrentLayerConfig,
        )
        self.assertEqual(
            attention.projection_strategy,
            SelfAttentionProjectionStrategy.SEPARATE,
        )
        logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
        self.assertEqual(
            tuple(logits.shape),
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_feed_forward_stack_flat_overrides_are_applied(self):
        cfg = self._baseline_config(
            {
                "ff_num_layers": 3,
                "ff_bias_flag": False,
                "ff_stack_hidden_dim": 24,
                "ff_stack_activation": ActivationOptions.MISH,
                "ff_stack_layer_norm_position": LayerNormPositionOptions.AFTER,
                "ff_stack_dropout_probability": 0.2,
                "ff_stack_last_layer_bias_option": LastLayerBiasOptions.DISABLED,
                "ff_stack_apply_output_pipeline_flag": False,
            }
        )
        stack = self._feed_forward_stack_config(cfg)
        self.assertIsInstance(stack, LayerStackConfig)
        self.assertEqual((stack.num_layers, stack.hidden_dim), (3, 24))
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

    def test_feed_forward_controls_attach_only_to_ff_stack(self):
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
        gate = stack.layer_config.gate_config.model_config
        self.assertEqual((gate.hidden_dim, gate.num_layers), (12, 3))
        self.assertIsNone(self._decoder_stack_config(cfg).layer_config.gate_config)
        self.assertIsNone(
            self._attention_projection_stack_config(cfg).layer_config.gate_config
        )

    def test_independent_feed_forward_controller_stacks_do_not_leak(self):
        defaults = self._default_builder_kwargs()
        stack_options = defaults["feed_forward_stack_options"]
        layer_options = defaults["feed_forward_layer_controller_options"]
        memory_options = defaults["feed_forward_dynamic_memory_options"]
        cfg = GptLinearConfigBuilder(
            **{
                **defaults,
                "feed_forward_stack_options": replace(
                    stack_options,
                    hidden_dim=17,
                ),
                "feed_forward_layer_controller_options": replace(
                    layer_options,
                    stack_gate_flag=True,
                    gate_stack_source=replace(
                        layer_options.gate_stack_source,
                        independent_flag=True,
                        hidden_dim=23,
                        num_layers=3,
                    ),
                    stack_halting_flag=True,
                    halting_stack_source=replace(
                        layer_options.halting_stack_source,
                        independent_flag=True,
                        hidden_dim=19,
                        num_layers=2,
                    ),
                ),
                "feed_forward_dynamic_memory_options": replace(
                    memory_options,
                    memory_flag=True,
                    memory_stack_source=replace(
                        memory_options.memory_stack_source,
                        independent_flag=True,
                        hidden_dim=31,
                        num_layers=4,
                    ),
                ),
            }
        ).build()
        stack = self._feed_forward_stack_config(cfg)
        gate = stack.layer_config.gate_config.model_config
        halting = stack.layer_config.halting_config.halting_gate_config
        memory = stack.shared_memory_config.model_config
        self.assertEqual(stack.hidden_dim, 17)
        self.assertEqual((gate.hidden_dim, gate.num_layers), (23, 3))
        self.assertEqual((halting.hidden_dim, halting.num_layers), (19, 2))
        self.assertEqual((memory.hidden_dim, memory.num_layers), (31, 4))
        self.assertIsNone(self._decoder_stack_config(cfg).layer_config.gate_config)
        self.assertIsNone(
            self._attention_projection_stack_config(cfg).layer_config.gate_config
        )

    def test_feed_forward_recurrence_forwards_batch(self):
        cfg = self._baseline_config(
            {
                "ff_recurrent_flag": True,
                "ff_recurrent_max_steps": 2,
            }
        )
        stack = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        self.assertIsInstance(stack, RecurrentLayerConfig)
        logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
        self.assertEqual(
            tuple(logits.shape),
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_module_entrypoint_resolves_cli_without_real_training(self):
        with (
            patch.object(sys, "argv", ["models.gpt.linear", "--preset", "baseline"]),
            patch(
                "models.package_cli.execute_runs",
                return_value=(),
            ) as execute_runs,
        ):
            runpy.run_module(
                "models.gpt.linear.__main__",
                run_name="__main__",
            )
        execute_runs.assert_called_once()
        package, plan = execute_runs.call_args.args
        self.assertEqual(package.catalog_key, "gpt/linear")
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

    def test_cli_and_config_listing_expose_only_gpt_boundary_flags(self):
        supported = {key.lower() for key in iter_supported_config_keys(config)}
        self.assertTrue(
            {
                "embedding_layer_norm_flag",
                "embedding_dropout_probability",
                "lm_head_bias_flag",
                "lm_head_weight_tying_flag",
            }
            <= supported
        )
        self.assertFalse(
            any(
                name.startswith(("mlm_", "nsp_"))
                or "token_type" in name
                or "causal" in name
                for name in supported
            )
        )
        parser = get_experiment_parser(
            ExperimentPreset.names(),
            "models.gpt.linear",
        )
        args = parser.parse_args(
            [
                "--preset",
                "baseline",
                "--input-dim",
                "32",
                "--output-dim",
                "32",
                "--hidden-dim",
                "16",
                "--sequence-length",
                "8",
                "--stack-num-layers",
                "2",
                "--attn-num-heads",
                "4",
                "--embedding-dropout-probability",
                "0.2",
                "--lm-head-bias-flag",
                "true",
                "--lm-head-weight-tying-flag",
                "false",
            ]
        )
        mode = resolve_experiment_mode(args, ExperimentPreset)
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides=mode.config_overrides,
        )[0]
        boundary = cfg.experiment_config.boundary_config
        self.assertEqual(boundary.embedding_options.dropout_probability, 0.2)
        self.assertTrue(boundary.lm_head_options.bias_flag)
        self.assertFalse(boundary.lm_head_options.weight_tying_flag)
        output = StringIO()
        with redirect_stdout(output):
            print_config_options("gpt/linear")
        listing = output.getvalue()
        self.assertIn("--embedding-dropout-probability", listing)
        self.assertIn("--lm-head-weight-tying-flag", listing)
        self.assertNotIn("--mlm-", listing)
        self.assertNotIn("--nsp-", listing)
        self.assertNotIn("--causal-attention-mask-flag", listing)

    def test_preset_contract_and_locks_cover_every_preset(self):
        presets = ExperimentPresets()
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, len(ExperimentPreset) + 1)),
        )
        self.assertNotIn("CAUSAL", ExperimentPreset.__members__)
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                overrides = presets.overrides_for_preset(preset)
                locks = presets.locks_for_preset(preset)
                self.assertEqual(set(locks), set(overrides))
                self.assertEqual(set(presets.locked_fields(preset)), set(locks))
                for field, lock in locks.items():
                    expected = overrides[field]
                    value = lock.value if isinstance(lock, PresetLock) else lock
                    self.assertEqual(value, expected)
                    self.assertIn(preset.name, lock.reason)

    def test_every_preset_forwards_both_datasets_causally(self):
        presets = ExperimentPresets()
        for dataset in self._default_datasets():
            for preset in ExperimentPreset:
                with self.subTest(dataset=dataset.__name__, preset=preset.name):
                    cfg = presets.get_config(
                        preset,
                        dataset,
                        config_overrides=self._preset_overrides(),
                    )[0]
                    logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
                    self.assertEqual(
                        tuple(logits.shape),
                        (2, cfg.sequence_length, dataset.num_classes),
                    )
                    self.assertEqual(tuple(auxiliary_loss.shape), ())
                    self.assertTrue(torch.isfinite(logits).all())
                    self.assertTrue(torch.isfinite(auxiliary_loss))
                    layer = self._decoder_layer_config(cfg)
                    self.assertTrue(layer.causal_attention_mask_flag)
                    self.assertTrue(
                        layer.self_attention_config.causal_attention_mask_flag
                    )
                    self.assertIsNone(layer.cross_attention_config)

    def test_every_preset_completes_one_tiny_training_epoch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = ExperimentPresets().get_config(
                    preset,
                    self._default_dataset(),
                    config_overrides=self._preset_overrides(),
                )[0]
                tiny_cpu_trainer().fit(
                    Model(cfg),
                    datamodule=RandomLanguageModelDataModule(
                        cfg,
                        batch_size=2,
                        num_batches=1,
                    ),
                )

    def test_random_search_and_unknown_axes(self):
        configs = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            RandomSearch(num_samples=2),
        )
        self.assertEqual(len(configs), 2)
        for search_key in (
            "bogus_axis",
            "decoder_options",
            "embedding_options",
        ):
            with self.subTest(search_key=search_key):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    ExperimentPresets().get_config(
                        ExperimentPreset.BASELINE,
                        self._default_dataset(),
                        RandomSearch(num_samples=2),
                        search_keys=[search_key],
                    )

    def test_grid_search_applies_decoder_axes(self):
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
            {self._decoder_stack_config(cfg).num_layers for cfg in configs},
            set(search_space.SEARCH_SPACE_STACK_NUM_LAYERS),
        )

    def test_flat_search_keys_are_supported(self):
        cases = {
            "layer_norm_position": (
                search_space.SEARCH_SPACE_LAYER_NORM_POSITION,
                lambda cfg: self._decoder_layer_config(cfg).layer_norm_position,
            ),
            "stack_layer_norm_position": (
                search_space.SEARCH_SPACE_STACK_LAYER_NORM_POSITION,
                lambda cfg: self._decoder_layer_config(cfg).layer_norm_position,
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

    def test_unlocked_flat_and_grouped_overrides_update_nested_config(self):
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides={
                "hidden_dim": 24,
                "sequence_length": 8,
                "stack_num_layers": 2,
                "stack_dropout_probability": 0.2,
                "attn_num_heads": 2,
                "ff_num_layers": 1,
                "embedding_dropout_probability": 0.3,
                "lm_head_weight_tying_flag": False,
                "lm_head_bias_flag": True,
            },
        )[0]
        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(cfg.sequence_length, 8)
        self.assertEqual(self._decoder_stack_config(cfg).num_layers, 2)
        self.assertEqual(self._decoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(self._attention_config(cfg).num_heads, 2)
        self.assertEqual(
            self._feed_forward_stack_config(cfg).num_layers,
            1,
        )
        boundary = cfg.experiment_config.boundary_config
        self.assertEqual(boundary.embedding_options.dropout_probability, 0.3)
        self.assertFalse(boundary.lm_head_options.weight_tying_flag)
        self.assertTrue(boundary.lm_head_options.bias_flag)

    def test_flat_overrides_take_precedence_over_grouped_bases(self):
        defaults = self._default_builder_kwargs()
        cfg = ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides={
                "input_dim": 32,
                "output_dim": 32,
                "decoder_options": replace(
                    defaults["decoder_options"],
                    hidden_dim=16,
                    num_layers=2,
                    dropout_probability=0.2,
                ),
                "embedding_options": replace(
                    defaults["embedding_options"],
                    dropout_probability=0.1,
                ),
                "hidden_dim": 24,
                "stack_num_layers": 3,
                "embedding_dropout_probability": 0.3,
            },
        )[0]
        self.assertEqual(cfg.hidden_dim, 24)
        self.assertEqual(self._decoder_stack_config(cfg).num_layers, 3)
        self.assertEqual(self._decoder_layer_config(cfg).dropout_probability, 0.2)
        self.assertEqual(
            cfg.experiment_config.boundary_config.embedding_options.dropout_probability,
            0.3,
        )

    def test_locked_presets_reject_override_and_search_conflicts(self):
        presets = ExperimentPresets()
        with self.assertRaisesRegex(ValueError, "PRE_NORM.*layer_norm_position"):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                self._default_dataset(),
                config_overrides={
                    "layer_norm_position": LayerNormPositionOptions.AFTER,
                },
            )
        with self.assertRaises(ValueError):
            presets.get_config(
                ExperimentPreset.PRE_NORM,
                self._default_dataset(),
                GridSearch(),
                search_keys=["layer_norm_position"],
            )

    def test_presets_wire_normalization_position_embedding_and_residuals(self):
        presets = ExperimentPresets()
        baseline = presets.get_config(ExperimentPreset.BASELINE)[0]
        self.assertIsInstance(
            baseline.experiment_config.positional_embedding_config,
            TextLearnedPositionalEmbeddingConfig,
        )
        self.assertTrue(self._decoder_layer_config(baseline).causal_attention_mask_flag)
        post_norm = presets.get_config(ExperimentPreset.POST_NORM)[0]
        self.assertEqual(
            self._decoder_layer_config(post_norm).layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        sinusoidal = presets.get_config(ExperimentPreset.SINUSOIDAL)[0]
        self.assertIsInstance(
            sinusoidal.experiment_config.positional_embedding_config,
            TextSinusoidalPositionalEmbeddingConfig,
        )
        attention_bias = presets.get_config(ExperimentPreset.ATTENTION_BIAS)[0]
        self.assertTrue(self._attention_config(attention_bias).add_key_value_bias_flag)
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
                stack = self._decoder_stack_config(cfg)
                self.assertEqual(
                    stack.layer_config.residual_connection_option,
                    ResidualConnectionOptions.RESIDUAL,
                )
                self.assertEqual(
                    isinstance(
                        cfg.experiment_config.decoder_config,
                        RecurrentLayerConfig,
                    ),
                    expected.get("recurrent", False),
                )
                self.assertEqual(
                    stack.layer_config.gate_config is not None,
                    expected.get("gate", False),
                )
                self.assertEqual(
                    stack.layer_config.halting_config is not None,
                    expected.get("halting", False),
                )
                self.assertEqual(
                    stack.shared_memory_config is not None,
                    expected.get("memory", False),
                )
                if expected.get("post_norm"):
                    self.assertEqual(
                        self._decoder_layer_config(cfg).layer_norm_position,
                        LayerNormPositionOptions.AFTER,
                    )

    def test_boundary_options_configure_embedding_and_lm_head(self):
        cfg = self._direct_config(
            embedding_options=GptEmbeddingOptions(
                layer_norm_flag=False,
                dropout_probability=0.25,
            ),
            lm_head_options=GptLmHeadOptions(
                weight_tying_flag=False,
                bias_flag=True,
            ),
        )
        model = Model(cfg)
        self.assertIsInstance(model, LanguageModelExperiment)
        self.assertIsInstance(model.embedding_layer_norm, nn.Identity)
        self.assertEqual(model.embedding_dropout.p, 0.25)
        self.assertIsNot(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNotNone(model.lm_head.bias)
        logits, auxiliary_loss = model(self._input_ids(cfg))
        self.assertEqual(
            tuple(logits.shape),
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_head_tying_is_conditional_and_untied_allows_vocab_mismatch(self):
        tied = Model(self._direct_config())
        untied = Model(
            self._direct_config(
                lm_head_options=GptLmHeadOptions(
                    weight_tying_flag=False,
                    bias_flag=False,
                )
            )
        )
        self.assertIs(tied.lm_head.weight, tied.token_embedding.weight)
        self.assertIsNot(untied.lm_head.weight, untied.token_embedding.weight)
        mismatch = self._direct_config(
            input_dim=29,
            output_dim=31,
            lm_head_options=GptLmHeadOptions(
                weight_tying_flag=False,
                bias_flag=False,
            ),
        )
        logits, _ = Model(mismatch)(self._input_ids(mismatch))
        self.assertEqual(logits.shape[-1], 31)
        with self.assertRaisesRegex(ValueError, "weight tying.*input_dim.*output_dim"):
            self._direct_config(input_dim=29, output_dim=31)

    def test_boundary_dimension_and_dropout_validation_matrix(self):
        defaults = self._default_builder_kwargs()
        cases = {
            "input_dim": {"input_dim": 0},
            "hidden_dim": {
                "decoder_options": replace(
                    defaults["decoder_options"],
                    hidden_dim=0,
                )
            },
            "output_dim": {
                "output_dim": 0,
                "lm_head_options": GptLmHeadOptions(
                    weight_tying_flag=False,
                    bias_flag=False,
                ),
            },
            "sequence_length": {"sequence_length": 0},
        }
        for field, overrides in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    GptLinearConfigBuilder(**overrides).build()
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "dropout_probability"):
                    GptLinearConfigBuilder(
                        embedding_options=GptEmbeddingOptions(
                            layer_norm_flag=True,
                            dropout_probability=probability,
                        )
                    ).build()

    def test_model_step_accepts_next_token_batch_and_backpropagates(self):
        cfg = self._direct_config()
        model = Model(cfg)
        inputs = self._input_ids(cfg)
        labels = torch.roll(inputs, shifts=-1, dims=1)
        output = model._model_step_outputs((inputs, labels))
        self.assertEqual(tuple(output.logits.shape), (2, 8, 32))
        self.assertEqual(tuple(output.total_loss.shape), ())
        self.assertTrue(torch.isfinite(output.total_loss))
        output.total_loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)

    def test_forward_converts_attention_mask_and_supplies_causal_mask(self):
        cfg = self._direct_config()
        model = Model(cfg)

        class SpyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None

            def forward(self, state):
                self.state = state
                state.loss = state.hidden.new_zeros(())
                return state

        spy = SpyDecoder()
        model.transformer = spy
        input_ids = self._input_ids(cfg)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -2:] = 0
        model(input_ids, attention_mask)
        self.assertIsInstance(spy.state, TransformerDecoderLayerState)
        torch.testing.assert_close(
            spy.state.target_key_padding_mask,
            attention_mask == 0,
        )
        expected = torch.triu(
            torch.ones(8, 8, dtype=torch.bool),
            diagonal=1,
        )
        torch.testing.assert_close(spy.state.target_attention_mask, expected)

    def test_decoder_auxiliary_loss_is_returned_and_added_to_total_loss(self):
        cfg = self._direct_config()
        model = Model(cfg)
        model.transformer = _AuxiliaryLossDecoder(0.25)
        inputs = self._input_ids(cfg)
        labels = torch.roll(inputs, shifts=-1, dims=1)
        output = model._model_step_outputs((inputs, labels))
        torch.testing.assert_close(
            output.auxiliary_loss,
            output.total_loss.new_tensor(0.25),
        )
        torch.testing.assert_close(
            output.total_loss,
            output.cross_entropy + output.auxiliary_loss,
        )

    def test_forward_validation_rejects_bad_ids_masks_and_contexts(self):
        model = Model(self.config())
        cases = (
            ((torch.tensor([1, 2]),), (ValueError,)),
            ((torch.tensor([[1.0, 2.0]]),), (TypeError,)),
            ((torch.tensor([[1, -1]]),), (ValueError,)),
            ((torch.tensor([[1, 16]]),), (ValueError,)),
            ((torch.ones((1, 7), dtype=torch.long),), (ValueError,)),
            (
                (
                    torch.tensor([[1, 2]]),
                    torch.ones((1, 3), dtype=torch.long),
                ),
                (ValueError,),
            ),
        )
        for arguments, errors in cases:
            with self.subTest(shapes=[tuple(value.shape) for value in arguments]):
                with self.assertRaises(errors):
                    model(*arguments)

    def test_decoder_is_built_from_decoder_block_layers(self):
        model = Model(self._direct_config())
        layers = list(model.transformer.layers)
        self.assertGreater(len(layers), 0)
        for layer in layers:
            with self.subTest(layer=type(layer).__name__):
                self.assertIsInstance(layer, TransformerDecoderBlockLayer)
                self.assertIsInstance(layer.model, TransformerDecoderLayer)
                self.assertIsNone(layer.model.cross_attention_model)

    def _default_builder_kwargs(self) -> dict:
        return linear_builder_kwargs_from_flat({}, config)

    def _baseline_config(self, overrides: dict | None = None):
        return ExperimentPresets().get_config(
            ExperimentPreset.BASELINE,
            self._default_dataset(),
            config_overrides={
                "batch_size": 2,
                "input_dim": 32,
                "output_dim": 32,
                "hidden_dim": 16,
                "sequence_length": 8,
                "stack_num_layers": 2,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 4,
                **(overrides or {}),
            },
        )[0]

    def _direct_config(
        self,
        *,
        input_dim: int = 32,
        output_dim: int = 32,
        embedding_options: GptEmbeddingOptions | None = None,
        lm_head_options: GptLmHeadOptions | None = None,
    ):
        runtime = runtime_from_flat(
            {
                "batch_size": 2,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "sequence_length": 8,
                "hidden_dim": 16,
                "stack_num_layers": 2,
                "stack_dropout_probability": 0.0,
                "attn_num_heads": 4,
            },
            config,
        )
        return GptLinearConfigBuilder(
            runtime=runtime,
            embedding_options=embedding_options,
            lm_head_options=lm_head_options,
        ).build()

    def _preset_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 8,
            "sequence_length": 4,
            "stack_num_layers": 2,
            "stack_dropout_probability": 0.0,
            "attn_num_heads": 2,
            "recurrent_max_steps": 2,
        }

    def _input_ids(self, cfg) -> torch.Tensor:
        return torch.randint(0, cfg.input_dim, (2, cfg.sequence_length))

    def _decoder_config(self, cfg):
        return cfg.experiment_config.decoder_config

    def _decoder_stack_config(self, cfg):
        decoder = self._decoder_config(cfg)
        if isinstance(decoder, RecurrentLayerConfig):
            return decoder.block_config
        return decoder

    def _decoder_layer_config(self, cfg):
        return self._decoder_stack_config(cfg).layer_config.layer_model_config

    def _attention_config(self, cfg):
        return self._decoder_layer_config(cfg).self_attention_config

    def _attention_projection_stack_config(self, cfg):
        projection = self._attention_config(cfg).projection_model_config
        if isinstance(projection, RecurrentLayerConfig):
            return projection.block_config
        return projection

    def _feed_forward_stack_config(self, cfg):
        stack = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        if isinstance(stack, RecurrentLayerConfig):
            return stack.block_config
        return stack

    def _decoder_gate_bias_flag(self, cfg) -> bool:
        gate = self._decoder_stack_config(cfg).layer_config.gate_config
        return gate.model_config.layer_config.layer_model_config.bias_flag

    def _default_datasets(self) -> list[type]:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _default_dataset(self) -> type:
        return self._default_datasets()[0]


class _AuxiliaryLossDecoder(nn.Module):
    def __init__(self, auxiliary_loss: float) -> None:
        super().__init__()
        self.auxiliary_loss = auxiliary_loss

    def forward(self, state: TransformerDecoderLayerState):
        state.loss = state.hidden.new_tensor(self.auxiliary_loss)
        return state


if __name__ == "__main__":
    unittest.main()
