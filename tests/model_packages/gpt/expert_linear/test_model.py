import importlib
import inspect
import unittest
from dataclasses import replace

import torch
import torch.nn as nn

import models.gpt.expert_linear.config as config
import models.gpt.expert_linear.dataset_options as dataset_options
import models.gpt.expert_linear.runtime_options as runtime_options
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.layers import (
    ActivationOptions,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import TransformerDecoderLayer, TransformerDecoderLayerState
from models.catalog import MODEL_CATALOG, catalog_entry
from models.gpt.expert_linear import (
    Experiment,
    ExperimentConfig,
    ExperimentPreset,
    ExperimentPresets,
    GptExpertLinearConfigBuilder,
    GptLmHeadOptions,
    Model,
)
from models.gpt.expert_linear._builder_adapter import (
    expert_linear_builder_kwargs_from_flat,
)
from models.gpt.expert_linear.runtime_options import (
    GptEmbeddingOptions,
    TransformerAttentionOptions,
    TransformerDecoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)
from models.training_test_utils import (
    RandomLanguageModelDataModule,
    tiny_cpu_trainer,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_SELF_ATTENTION_TYPE = SelfAttentionConfig().registry_owner()


class TestGptExpertLinearModel(unittest.TestCase):
    def test_runtime_defaults_describe_a_gpt2_decoder_block(self):
        self.assertEqual(
            config.STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.BEFORE,
        )
        self.assertFalse(config.EMBEDDING_LAYER_NORM_FLAG)
        self.assertEqual(config.FF_NUM_LAYERS, 1)
        self.assertEqual(config.ATTN_STACK_ACTIVATION, ActivationOptions.DISABLED)
        self.assertFalse(config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG)
        self.assertEqual(
            config.FF_STACK_LAYER_NORM_POSITION,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertFalse(config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG)

    backend_module_name = "MixtureOfExperts"

    def config(self, **overrides):
        return GptExpertLinearConfigBuilder(
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
            MODEL_CATALOG["gpt/expert_linear"].module_path,
            "models.gpt.expert_linear",
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
            GptExpertLinearConfigBuilder(
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
            rtol=1e-5,
            atol=1e-5,
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
        package = importlib.import_module("models.gpt.expert_linear")
        self.assertIn("GptExpertLinearConfigBuilder", package.__all__)
        for module_name in (
            "models.gpt.expert_linear.config",
            "models.gpt.expert_linear.presets",
            "models.gpt.expert_linear.model",
            "models.gpt.expert_linear.config_builder",
            "models.gpt.expert_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                self.assertEqual(
                    importlib.import_module(module_name).__name__,
                    module_name,
                )
        self.assertEqual(Experiment()._public_model_id(), "gpt/expert_linear")
        self.assertIsNotNone(catalog_entry("gpt/expert_linear"))

    def test_attention_mode_and_causal_switches_are_not_public(self):
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(ExperimentPreset, "EXPERT_ATTENTION"))
        self.assertFalse(hasattr(config, "CAUSAL_ATTENTION_MASK_FLAG"))
        self.assertNotIn("CAUSAL", ExperimentPreset.__members__)
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 22)),
        )
        parameters = inspect.signature(GptExpertLinearConfigBuilder).parameters
        self.assertNotIn("expert_attention_flag", parameters)
        self.assertNotIn("causal_attention_mask_flag", parameters)
        self.assertIn("expert_attention_use_kv_expert_models_flag", parameters)
        with self.assertRaises(TypeError):
            GptExpertLinearConfigBuilder(expert_attention_flag=False)
        with self.assertRaises(TypeError):
            GptExpertLinearConfigBuilder(causal_attention_mask_flag=False)

    def test_runtime_options_are_non_adaptive_and_gpt_specific(self):
        names = {
            name
            for name, value in vars(runtime_options).items()
            if inspect.isclass(value) and value.__module__ == runtime_options.__name__
        }
        self.assertFalse(any("Adaptive" in name for name in names))
        self.assertIn("GptEmbeddingOptions", names)
        self.assertIn("GptLmHeadOptions", names)
        self.assertIn("ExpertsMixtureOptions", names)
        self.assertNotIn("BertEmbeddingOptions", names)

    def test_flat_options_build_same_config_as_grouped_options(self):
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
        flat_kwargs = {
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
            "num_experts": 4,
            "top_k": 2,
            "expert_stack_hidden_dim": 12,
            "router_stack_hidden_dim": 10,
        }
        adapted = expert_linear_builder_kwargs_from_flat(flat_kwargs, config)
        self.assertEqual(adapted["embedding_options"], embedding_options)
        self.assertEqual(adapted["decoder_options"], decoder_options)
        self.assertEqual(adapted["positional_embedding_options"], positional_options)
        self.assertEqual(adapted["attention_options"], attention_options)
        self.assertEqual(adapted["feed_forward_options"], feed_forward_options)
        self.assertEqual(adapted["lm_head_options"], lm_head_options)
        flat_config = GptExpertLinearConfigBuilder(**adapted).build()
        grouped_config = GptExpertLinearConfigBuilder(
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
            mixture_options=adapted["mixture_options"],
            expert_stack_options=adapted["expert_stack_options"],
            router_stack_options=adapted["router_stack_options"],
        ).build()
        self.assertEqual(flat_config, grouped_config)

    def test_feed_forward_stack_is_expert_backed(self):
        cfg = self._preset_config(ExperimentPreset.TOP1_SWITCH_AUX)
        mixture = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        expert_core = mixture.stack_config.layer_config.layer_model_config
        self.assertIsInstance(mixture, MixtureOfExpertsModelConfig)
        self.assertEqual(mixture.top_k, 1)
        self.assertEqual(mixture.sampler_config.switch_loss_weight, 0.1)
        self.assertIsInstance(
            expert_core.expert_model_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_outer_and_inner_expert_controls_are_independent(self):
        defaults = self._default_builder_kwargs()
        for name, outer_gate, inner_gate in (
            ("outer", True, False),
            ("inner", False, True),
        ):
            with self.subTest(name=name):
                cfg = self._preset_config(
                    ExperimentPreset.BASELINE,
                    {
                        "feed_forward_layer_controller_options": replace(
                            defaults["feed_forward_layer_controller_options"],
                            stack_gate_flag=outer_gate,
                        ),
                        "expert_layer_controller_options": replace(
                            defaults["expert_layer_controller_options"],
                            stack_gate_flag=inner_gate,
                        ),
                    },
                )
                mixture = self._decoder_layer_config(
                    cfg
                ).feed_forward_config.stack_config
                outer = mixture.stack_config
                inner = outer.layer_config.layer_model_config.expert_model_config
                self.assertEqual(
                    outer.layer_config.gate_config is not None,
                    outer_gate,
                )
                self.assertEqual(
                    inner.layer_config.gate_config is not None,
                    inner_gate,
                )

    def test_feed_forward_recurrence_wraps_outer_mixture(self):
        defaults = self._default_builder_kwargs()
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "feed_forward_recurrent_controller_options": replace(
                    defaults["feed_forward_recurrent_controller_options"],
                    recurrent_flag=True,
                    recurrent_max_steps=2,
                )
            },
        )
        feed_forward = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        logits, _ = Model(cfg)(self._input_ids(cfg))
        self.assertIsInstance(feed_forward, RecurrentLayerConfig)
        self.assertIsInstance(feed_forward.block_config, MixtureOfExpertsModelConfig)
        self.assertEqual(
            tuple(logits.shape),
            (2, cfg.sequence_length, cfg.output_dim),
        )

    def test_attention_controls_apply_only_to_regular_projection_path(self):
        defaults = self._default_builder_kwargs()
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "attention_projection_stack_options": replace(
                    defaults["attention_projection_stack_options"],
                    hidden_dim=17,
                ),
                "attention_projection_layer_controller_options": replace(
                    defaults["attention_projection_layer_controller_options"],
                    stack_gate_flag=True,
                ),
                "expert_stack_options": replace(
                    defaults["expert_stack_options"],
                    hidden_dim=11,
                ),
            },
        )
        attention = self._decoder_layer_config(cfg).self_attention_config
        projection = attention.projection_model_config
        experts = attention.experts_config.expert_model_config
        self.assertIsInstance(attention, MixtureOfAttentionHeadsConfig)
        self.assertEqual(projection.hidden_dim, 17)
        self.assertIsNotNone(projection.layer_config.gate_config)
        self.assertEqual(experts.hidden_dim, 11)
        self.assertIsNone(experts.layer_config.gate_config)

    def test_every_preset_executes_mixture_attention_and_expert_ff(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._preset_config(preset)
                model = Model(cfg)
                logits, auxiliary_loss = model(self._input_ids(cfg))
                modules = tuple(model.modules())
                self.assertTrue(
                    any(isinstance(m, _MIXTURE_ATTENTION_TYPE) for m in modules)
                )
                self.assertTrue(
                    any(isinstance(m, _MIXTURE_OF_EXPERTS_TYPE) for m in modules)
                )
                self.assertFalse(
                    any(isinstance(m, _SELF_ATTENTION_TYPE) for m in modules)
                )
                self.assertEqual(
                    tuple(logits.shape),
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(tuple(auxiliary_loss.shape), ())
                self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_decoder_layers_own_disjoint_attention_and_ff_experts(self):
        defaults = self._default_builder_kwargs()
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "decoder_options": replace(
                    defaults["decoder_options"],
                    hidden_dim=8,
                    num_layers=2,
                    dropout_probability=0.0,
                )
            },
        )
        model = Model(cfg)
        decoder_layers = [layer.model for layer in model.transformer.layers]
        attentions = [layer.self_attention_model for layer in decoder_layers]
        feed_forwards = [layer.feed_forward_model for layer in decoder_layers]
        self.assertEqual(len(decoder_layers), 2)
        for left, right in (
            (attentions[0], attentions[1]),
            (feed_forwards[0], feed_forwards[1]),
            (attentions[0], feed_forwards[0]),
            (attentions[1], feed_forwards[1]),
        ):
            self._assert_parameter_sets_are_disjoint(left, right)

    def test_attention_kv_modes_forward_backward(self):
        cases = (
            ("regular_kv", ExperimentPreset.BASELINE, False, False),
            ("expert_kv", ExperimentPreset.BASELINE, True, False),
            ("expert_kv_bias", ExperimentPreset.ATTENTION_BIAS, True, True),
        )
        for name, preset, use_kv_experts, expect_bias in cases:
            with self.subTest(name=name):
                torch.manual_seed(0)
                cfg = self._preset_config(
                    preset,
                    {"expert_attention_use_kv_expert_models_flag": (use_kv_experts)},
                )
                model = Model(cfg)
                logits, auxiliary_loss = model(self._input_ids(cfg))
                attention = next(
                    m for m in model.modules() if isinstance(m, _MIXTURE_ATTENTION_TYPE)
                )
                self.assertEqual(
                    attention.cfg.use_kv_expert_models_flag,
                    use_kv_experts,
                )
                self.assertEqual(
                    attention.cfg.add_key_value_bias_flag,
                    expect_bias,
                )
                self.assertTrue(torch.isfinite(auxiliary_loss))
                (logits.square().mean() + auxiliary_loss).backward()
                self._assert_nonzero_parameter_gradients(
                    model.token_embedding,
                    "token embedding",
                )
                projector = attention.projector
                if use_kv_experts:
                    for role, expert_model in {
                        "query": projector.query_model,
                        "key": projector.key_model,
                        "value": projector.value_model,
                        "output": projector.output_model,
                    }.items():
                        with self.subTest(name=name, role=role):
                            self.assertIsInstance(
                                expert_model, _MIXTURE_OF_EXPERTS_LAYER_TYPE
                            )
                            self._assert_nonzero_parameter_gradients(
                                expert_model,
                                role,
                            )
                else:
                    self._assert_nonzero_parameter_gradients(
                        projector.key_model,
                        "regular key projection",
                    )
                    self._assert_nonzero_parameter_gradients(
                        projector.value_model,
                        "regular value projection",
                    )
                if expect_bias:
                    self.assertGreater(
                        attention.bias.key_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )
                    self.assertGreater(
                        attention.bias.value_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )

    def test_routed_auxiliary_loss_is_added_to_language_model_loss(self):
        torch.manual_seed(0)
        cfg = self._preset_config(ExperimentPreset.TOP1_SWITCH_AUX)
        model = Model(cfg)
        inputs = self._input_ids(cfg)
        labels = torch.roll(inputs, shifts=-1, dims=1)
        output = model._model_step_outputs((inputs, labels))
        self.assertGreater(output.auxiliary_loss.item(), 0.0)
        torch.testing.assert_close(
            output.total_loss,
            output.cross_entropy + output.auxiliary_loss,
        )

    def test_forward_runs_one_decoder_pass_and_leaves_causality_to_attention(self):
        cfg = self._preset_config(ExperimentPreset.BASELINE)
        model = Model(cfg)

        class SpyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None
                self.calls = 0

            def forward(self, state):
                self.calls += 1
                self.state = state
                state.loss = state.hidden.new_zeros(())
                return state

        spy = SpyDecoder()
        model.transformer = spy
        input_ids = self._input_ids(cfg)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -1] = 0
        model(input_ids, attention_mask)
        self.assertEqual(spy.calls, 1)
        self.assertIsInstance(spy.state, TransformerDecoderLayerState)
        torch.testing.assert_close(
            spy.state.target_key_padding_mask,
            attention_mask == 0,
        )
        self.assertIsNone(spy.state.target_attention_mask)

    def test_boundary_options_and_tying_are_fully_configurable(self):
        defaults = self._default_builder_kwargs()
        cfg = self._direct_config(
            embedding_options=GptEmbeddingOptions(
                layer_norm_flag=False,
                dropout_probability=0.25,
            ),
            lm_head_options=replace(
                defaults["lm_head_options"],
                weight_tying_flag=False,
                bias_flag=True,
            ),
        )
        model = Model(cfg)
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

    def test_untied_head_allows_mismatched_vocabularies(self):
        defaults = self._default_builder_kwargs()
        cfg = self._direct_config(
            input_dim=29,
            output_dim=31,
            lm_head_options=replace(
                defaults["lm_head_options"],
                weight_tying_flag=False,
            ),
        )
        logits, _ = Model(cfg)(self._input_ids(cfg))
        self.assertEqual(logits.shape[-1], 31)
        with self.assertRaisesRegex(ValueError, "weight tying.*input_dim.*output_dim"):
            self._direct_config(input_dim=29, output_dim=31)

    def test_invalid_dimensions_and_dropout_are_rejected(self):
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
                "lm_head_options": replace(
                    defaults["lm_head_options"],
                    weight_tying_flag=False,
                ),
            },
            "sequence_length": {"sequence_length": 0},
        }
        for field, overrides in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self._direct_config(**overrides)
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "dropout_probability"):
                    self._direct_config(
                        embedding_options=replace(
                            defaults["embedding_options"],
                            dropout_probability=probability,
                        )
                    )

    def test_baseline_forwards_both_language_model_datasets(self):
        for dataset in dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]:
            with self.subTest(dataset=dataset.__name__):
                overrides = self._small_overrides()
                overrides.pop("input_dim")
                overrides.pop("output_dim")
                cfg = ExperimentPresets().get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides=overrides,
                )[0]
                logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
                self.assertEqual(logits.shape[-1], dataset.num_classes)
                self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_representative_presets_train_tiny_epochs(self):
        for preset in (
            ExperimentPreset.BASELINE,
            ExperimentPreset.TOP1_SWITCH_AUX,
        ):
            with self.subTest(preset=preset.name):
                cfg = self._preset_config(preset)
                tiny_cpu_trainer().fit(
                    Model(cfg),
                    datamodule=RandomLanguageModelDataModule(
                        cfg,
                        batch_size=2,
                        num_batches=1,
                    ),
                )

    def _preset_config(
        self,
        preset: ExperimentPreset,
        overrides: dict | None = None,
    ):
        return ExperimentPresets().get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides={
                **self._small_overrides(),
                **(overrides or {}),
            },
        )[0]

    def _direct_config(self, **overrides):
        kwargs = expert_linear_builder_kwargs_from_flat(
            self._small_overrides(),
            config,
        )
        kwargs.update(overrides)
        return GptExpertLinearConfigBuilder(**kwargs).build()

    def _small_overrides(self) -> dict:
        defaults = self._default_builder_kwargs()
        return {
            "batch_size": 2,
            "input_dim": 32,
            "output_dim": 32,
            "sequence_length": 6,
            "decoder_options": replace(
                defaults["decoder_options"],
                hidden_dim=8,
                num_layers=2,
                dropout_probability=0.0,
            ),
            "attention_options": replace(
                defaults["attention_options"],
                num_heads=2,
            ),
            "attention_projection_stack_options": replace(
                defaults["attention_projection_stack_options"],
                hidden_dim=8,
            ),
            "feed_forward_stack_options": replace(
                defaults["feed_forward_stack_options"],
                hidden_dim=8,
            ),
            "submodule_stack_options": replace(
                defaults["submodule_stack_options"],
                hidden_dim=8,
            ),
            "mixture_options": replace(
                defaults["mixture_options"],
                num_experts=4,
            ),
            "expert_stack_options": replace(
                defaults["expert_stack_options"],
                hidden_dim=8,
            ),
            "router_stack_options": replace(
                defaults["router_stack_options"],
                hidden_dim=8,
            ),
            "recurrent_controller_options": replace(
                defaults["recurrent_controller_options"],
                recurrent_max_steps=2,
            ),
        }

    def _default_builder_kwargs(self) -> dict:
        return expert_linear_builder_kwargs_from_flat({}, config)

    def _input_ids(self, cfg) -> torch.Tensor:
        return torch.randint(0, cfg.input_dim, (2, cfg.sequence_length))

    def _decoder_layer_config(self, cfg):
        decoder = cfg.experiment_config.decoder_config
        if isinstance(decoder, RecurrentLayerConfig):
            decoder = decoder.block_config
        return decoder.layer_config.layer_model_config

    def _assert_nonzero_parameter_gradients(self, model, role: str) -> None:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients, f"{role} model had no gradients")
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in gradients),
            f"{role} model gradients were all zero",
        )

    def _assert_parameter_sets_are_disjoint(self, left, right) -> None:
        left_ids = {id(parameter) for parameter in left.parameters()}
        right_ids = {id(parameter) for parameter in right.parameters()}
        self.assertTrue(left_ids)
        self.assertTrue(right_ids)
        self.assertTrue(left_ids.isdisjoint(right_ids))


if __name__ == "__main__":
    unittest.main()
