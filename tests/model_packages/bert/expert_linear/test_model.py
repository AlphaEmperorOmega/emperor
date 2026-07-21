import importlib
import inspect
import unittest
from dataclasses import replace

import torch
from torch import nn

import models.bert.expert_linear.config as config
import models.bert.expert_linear.dataset_options as dataset_options
import models.bert.expert_linear.runtime_options as runtime_options
from emperor.attention import (
    AttentionLayerState,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.bert_pretraining import BertPretrainingExperiment
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.layers import (
    ActivationOptions,
    LayerNormPositionOptions,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from models.bert.expert_linear._builder_adapter import (
    expert_linear_builder_kwargs_from_flat,
)
from models.bert.expert_linear.config_builder import (
    BertExpertLinearConfigBuilder,
)
from models.bert.expert_linear.model import Model
from models.bert.expert_linear.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.bert.expert_linear.runtime_options import (
    BertEmbeddingOptions,
    BertMlmHeadOptions,
    BertNspHeadOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
    TransformerPositionalEmbeddingOptions,
)
from models.catalog import catalog_entry
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_SELF_ATTENTION_TYPE = SelfAttentionConfig().registry_owner()


def _default_builder_kwargs() -> dict:
    return expert_linear_builder_kwargs_from_flat({}, config)


class TestBertExpertLinearModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        package = importlib.import_module("models.bert.expert_linear")
        self.assertEqual(package.__all__, ["Experiment", "ExperimentPreset"])

        for module_name in (
            "models.bert.expert_linear.config",
            "models.bert.expert_linear.presets",
            "models.bert.expert_linear.model",
            "models.bert.expert_linear.config_builder",
            "models.bert.expert_linear.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.__name__, module_name)

        self.assertEqual(Experiment()._public_model_id(), "bert/expert_linear")
        self.assertIsNotNone(catalog_entry("bert/expert_linear"))

    def test_attention_mode_switch_is_removed(self):
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(ExperimentPreset, "EXPERT_ATTENTION"))
        self.assertEqual(
            [preset.value for preset in ExperimentPreset], list(range(1, 23))
        )
        self.assertNotIn(
            "expert_attention_flag",
            inspect.signature(BertExpertLinearConfigBuilder).parameters,
        )

        with self.assertRaises(TypeError):
            BertExpertLinearConfigBuilder(expert_attention_flag=False)
        for legacy_kwargs in (
            {"hidden_dim": 16},
            {"embedding_dropout_probability": 0.2},
            {"mlm_decoder_bias_flag": False},
        ):
            with self.subTest(legacy_kwargs=legacy_kwargs):
                with self.assertRaises(TypeError):
                    BertExpertLinearConfigBuilder(**legacy_kwargs)
        with self.assertRaises(TypeError):
            self._config(
                ExperimentPreset.BASELINE,
                {"expert_attention_flag": False},
            )

    def test_runtime_options_are_non_adaptive_and_canonical(self):
        names = {
            name
            for name, value in vars(runtime_options).items()
            if inspect.isclass(value) and value.__module__ == runtime_options.__name__
        }

        self.assertFalse(any("Adaptive" in name for name in names))
        self.assertIn("BertEmbeddingOptions", names)
        self.assertIn("ExpertsMixtureOptions", names)

    def test_flat_options_build_the_same_config_as_grouped_options(self):
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
            "output_dim": 32,
            "sequence_length": 8,
            "token_type_vocab_size": 4,
            "embedding_layer_norm_flag": False,
            "embedding_dropout_probability": 0.2,
            "hidden_dim": 16,
            "stack_num_layers": 1,
            "stack_activation": ActivationOptions.MISH,
            "stack_dropout_probability": 0.1,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
            "causal_attention_mask_flag": True,
            "positional_embedding_option": TextSinusoidalPositionalEmbeddingConfig,
            "positional_embedding_padding_idx": 0,
            "positional_embedding_auto_expand_flag": True,
            "attn_num_heads": 4,
            "attn_num_layers": 2,
            "attn_bias_flag": True,
            "attn_add_key_value_bias_flag": True,
            "ff_num_layers": 2,
            "ff_bias_flag": False,
            "mlm_activation": ActivationOptions.SILU,
            "mlm_dense_bias_flag": False,
            "mlm_layer_norm_flag": False,
            "mlm_decoder_bias_flag": False,
            "mlm_decoder_weight_tying_flag": True,
            "nsp_pooler_activation": ActivationOptions.SIGMOID,
            "nsp_pooler_bias_flag": False,
            "nsp_output_dim": 3,
            "nsp_head_bias_flag": False,
            "num_experts": 4,
            "top_k": 2,
            "expert_stack_hidden_dim": 12,
            "router_stack_hidden_dim": 10,
        }
        adapted = expert_linear_builder_kwargs_from_flat(flat_kwargs, config)

        self.assertEqual(adapted["embedding_options"], embedding_options)
        self.assertEqual(adapted["encoder_options"], encoder_options)
        self.assertEqual(
            adapted["positional_embedding_options"],
            positional_embedding_options,
        )
        self.assertEqual(adapted["attention_options"], attention_options)
        self.assertEqual(adapted["feed_forward_options"], feed_forward_options)
        self.assertEqual(adapted["mlm_head_options"], mlm_head_options)
        self.assertEqual(adapted["nsp_head_options"], nsp_head_options)
        self.assertNotIn("embedding_dropout_probability", adapted)

        flat_config = BertExpertLinearConfigBuilder(**adapted).build()
        grouped_config = BertExpertLinearConfigBuilder(
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
            mixture_options=adapted["mixture_options"],
            expert_stack_options=adapted["expert_stack_options"],
            router_stack_options=adapted["router_stack_options"],
        ).build()

        self.assertEqual(flat_config, grouped_config)

    def test_feed_forward_stack_is_expert_backed(self):
        cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
        feed_forward_stack_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        expert_core_config = (
            feed_forward_stack_config.stack_config.layer_config.layer_model_config
        )

        self.assertIsInstance(feed_forward_stack_config, MixtureOfExpertsModelConfig)
        self.assertEqual(feed_forward_stack_config.top_k, 1)
        self.assertEqual(
            feed_forward_stack_config.sampler_config.switch_loss_weight,
            0.1,
        )
        self.assertIsInstance(
            expert_core_config.expert_model_config.layer_config.layer_model_config,
            LinearLayerConfig,
        )

    def test_outer_and_inner_expert_controls_are_independent(self):
        cases = (
            ("outer", True, False),
            ("inner", False, True),
        )
        defaults = _default_builder_kwargs()
        for name, outer_gate, inner_gate in cases:
            with self.subTest(name=name):
                cfg = self._config(
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
                mixture_config = self._encoder_layer_config(
                    cfg
                ).feed_forward_config.stack_config
                outer_stack = mixture_config.stack_config
                inner_stack = (
                    outer_stack.layer_config.layer_model_config.expert_model_config
                )

                self.assertEqual(
                    outer_stack.layer_config.gate_config is not None,
                    outer_gate,
                )
                self.assertEqual(
                    inner_stack.layer_config.gate_config is not None,
                    inner_gate,
                )

    def test_feed_forward_recurrence_wraps_the_outer_mixture(self):
        defaults = _default_builder_kwargs()
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                "feed_forward_recurrent_controller_options": replace(
                    defaults["feed_forward_recurrent_controller_options"],
                    recurrent_flag=True,
                    recurrent_max_steps=2,
                )
            },
        )
        feed_forward_config = self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config
        mlm_logits, nsp_logits, _ = Model(cfg)(*self._fake_bert_inputs(cfg))

        self.assertIsInstance(feed_forward_config, RecurrentLayerConfig)
        self.assertIsInstance(
            feed_forward_config.block_config,
            MixtureOfExpertsModelConfig,
        )
        self.assertEqual(
            mlm_logits.shape,
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(nsp_logits.shape, (2, 2))

    def test_attention_controls_apply_only_to_regular_projection_stack(self):
        defaults = _default_builder_kwargs()
        cfg = self._config(
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
        attention_config = self._encoder_layer_config(cfg).attention_config
        projection_stack = attention_config.projection_model_config
        expert_stack = attention_config.experts_config.expert_model_config

        self.assertIsInstance(attention_config, MixtureOfAttentionHeadsConfig)
        self.assertEqual(projection_stack.hidden_dim, 17)
        self.assertIsNotNone(projection_stack.layer_config.gate_config)
        self.assertEqual(expert_stack.hidden_dim, 11)
        self.assertIsNone(expert_stack.layer_config.gate_config)

    def test_all_presets_build_and_execute_mixture_of_attention_heads(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                model = Model(cfg)
                mlm_logits, nsp_logits, auxiliary_loss = model(
                    *self._fake_bert_inputs(cfg)
                )
                modules = tuple(model.modules())

                self.assertTrue(
                    any(
                        isinstance(module, _MIXTURE_ATTENTION_TYPE)
                        for module in modules
                    )
                )
                self.assertFalse(
                    any(isinstance(module, _SELF_ATTENTION_TYPE) for module in modules)
                )
                self.assertEqual(
                    mlm_logits.shape,
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_encoder_layers_have_disjoint_attention_and_feed_forward_experts(self):
        defaults = _default_builder_kwargs()
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                "encoder_options": replace(
                    defaults["encoder_options"],
                    hidden_dim=8,
                    num_layers=2,
                    dropout_probability=0.0,
                )
            },
        )
        model = Model(cfg)
        encoder_layers = [layer.model for layer in model.transformer.layers]
        attentions = [layer.self_attention_model for layer in encoder_layers]
        feed_forwards = [layer.feed_forward_model for layer in encoder_layers]

        self.assertEqual(len(encoder_layers), 2)
        self.assertTrue(
            all(
                isinstance(attention, _MIXTURE_ATTENTION_TYPE)
                for attention in attentions
            )
        )
        for left, right in (
            (attentions[0], attentions[1]),
            (feed_forwards[0], feed_forwards[1]),
            (attentions[0], feed_forwards[0]),
            (attentions[1], feed_forwards[1]),
        ):
            with self.subTest(
                left=type(left).__name__,
                right=type(right).__name__,
            ):
                self._assert_parameter_sets_are_disjoint(left, right)

    def test_attention_kv_modes_forward_backward(self):
        cases = (
            ("regular_kv", ExperimentPreset.BASELINE, False, False),
            ("expert_kv", ExperimentPreset.BASELINE, True, False),
            ("expert_kv_with_bias", ExperimentPreset.ATTENTION_BIAS, True, True),
        )
        for name, preset, use_kv_experts, expect_bias in cases:
            with self.subTest(name=name):
                torch.manual_seed(0)
                cfg = self._config(
                    preset,
                    {"expert_attention_use_kv_expert_models_flag": (use_kv_experts)},
                )
                model = Model(cfg)
                mlm_logits, nsp_logits, auxiliary_loss = model(
                    *self._fake_bert_inputs(cfg)
                )

                self.assertEqual(
                    mlm_logits.shape,
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertGreater(auxiliary_loss.item(), 0.0)
                self.assertTrue(auxiliary_loss.requires_grad)

                attention = next(
                    module
                    for module in model.modules()
                    if isinstance(module, _MIXTURE_ATTENTION_TYPE)
                )
                self.assertEqual(
                    attention.cfg.use_kv_expert_models_flag,
                    use_kv_experts,
                )
                self.assertEqual(
                    attention.cfg.add_key_value_bias_flag,
                    expect_bias,
                )

                loss = mlm_logits.square().mean() + nsp_logits.square().mean()
                loss = loss + auxiliary_loss
                loss.backward()
                self.assertIsNotNone(model.token_embedding.weight.grad)
                self.assertGreater(
                    model.token_embedding.weight.grad.abs().sum().item(),
                    0.0,
                )

                projector = attention.projector
                if use_kv_experts:
                    expert_models = {
                        "query": projector.query_model,
                        "key": projector.key_model,
                        "value": projector.value_model,
                        "output": projector.output_model,
                    }
                    for role, expert_model in expert_models.items():
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
                    self.assertIsNotNone(attention.bias.key_bias_vector.grad)
                    self.assertIsNotNone(attention.bias.value_bias_vector.grad)
                    self.assertGreater(
                        attention.bias.key_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )
                    self.assertGreater(
                        attention.bias.value_bias_vector.grad.abs().sum().item(),
                        0.0,
                    )

    def test_routed_auxiliary_loss_is_added_to_pretraining_loss(self):
        torch.manual_seed(0)
        cfg = self._config(ExperimentPreset.TOP1_SWITCH_AUX)
        model = Model(cfg)
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(cfg)
        mlm_labels = torch.full_like(input_ids, -100)
        mlm_labels[:, 1] = input_ids[:, 1]
        next_sentence_labels = torch.tensor([0, 1])

        output = model._model_step_outputs(
            (
                input_ids,
                mlm_labels,
                attention_mask,
                token_type_ids,
                next_sentence_labels,
            )
        )

        self.assertGreater(output.auxiliary_loss.item(), 0.0)
        torch.testing.assert_close(
            output.total_loss,
            output.mlm_loss + output.nsp_loss + output.auxiliary_loss,
        )

    def test_causal_mask_is_propagated_and_blocks_future_tokens(self):
        cfg = self._config(ExperimentPreset.CAUSAL)
        attention_config = self._encoder_layer_config(cfg).attention_config
        self.assertTrue(attention_config.causal_attention_mask_flag)

        model = Model(cfg).eval()
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(cfg)
        changed = input_ids.clone()
        changed[:, -2] = (changed[:, -2] + 17) % cfg.input_dim
        with torch.no_grad():
            original_mlm, original_nsp, _ = model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            changed_mlm, changed_nsp, _ = model(
                changed,
                attention_mask,
                token_type_ids,
            )

        torch.testing.assert_close(original_mlm[:, 0], changed_mlm[:, 0])
        torch.testing.assert_close(original_nsp, changed_nsp)

    def test_key_padding_mask_is_forwarded_to_the_encoder(self):
        cfg = self._config(ExperimentPreset.BASELINE)
        model = Model(cfg)

        class SpyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None

            def forward(self, state):
                self.state = state
                state.loss = state.hidden.new_zeros(())
                return state

        spy = SpyEncoder()
        model.transformer = spy
        input_ids, attention_mask, token_type_ids = self._fake_bert_inputs(cfg)

        model(input_ids, attention_mask, token_type_ids)

        self.assertIsInstance(spy.state, AttentionLayerState)
        torch.testing.assert_close(
            spy.state.key_padding_mask,
            attention_mask == 0,
        )
        self.assertIsNone(spy.state.attention_mask)

    def test_boundary_options_configure_embeddings_and_heads(self):
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

        self.assertIsInstance(model, BertPretrainingExperiment)
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

        mlm_logits, nsp_logits, auxiliary_loss = model(*self._fake_bert_inputs(cfg))
        self.assertEqual(
            mlm_logits.shape,
            (2, cfg.sequence_length, cfg.output_dim),
        )
        self.assertEqual(nsp_logits.shape, (2, 3))
        self.assertEqual(auxiliary_loss.dim(), 0)

    def test_mlm_decoder_tying_is_conditional(self):
        tied = Model(self._direct_config())
        untied = Model(
            self._direct_config(
                mlm_head_options=replace(
                    _default_builder_kwargs()["mlm_head_options"],
                    decoder_weight_tying_flag=False,
                )
            )
        )

        self.assertIs(tied.mlm_decoder.weight, tied.token_embedding.weight)
        self.assertIsNot(untied.mlm_decoder.weight, untied.token_embedding.weight)

        mismatched = self._direct_config(
            input_dim=29,
            output_dim=31,
            mlm_head_options=replace(
                _default_builder_kwargs()["mlm_head_options"],
                decoder_weight_tying_flag=False,
            ),
        )
        mlm_logits, _, _ = Model(mismatched)(*self._fake_bert_inputs(mismatched))
        self.assertEqual(mlm_logits.shape[-1], 31)

        with self.assertRaisesRegex(ValueError, "weight tying.*input_dim.*output_dim"):
            self._direct_config(input_dim=29, output_dim=31)

    def test_invalid_dimensions_and_dropout_are_rejected(self):
        defaults = _default_builder_kwargs()
        cases = {
            "input_dim": {"input_dim": 0},
            "hidden_dim": {
                "encoder_options": replace(
                    defaults["encoder_options"],
                    hidden_dim=0,
                )
            },
            "output_dim": {
                "output_dim": 0,
                "mlm_head_options": replace(
                    defaults["mlm_head_options"],
                    decoder_weight_tying_flag=False,
                ),
            },
            "sequence_length": {"sequence_length": 0},
            "token_type_vocab_size": {
                "embedding_options": replace(
                    defaults["embedding_options"],
                    token_type_vocab_size=0,
                )
            },
            "nsp_output_dim": {
                "nsp_head_options": replace(
                    defaults["nsp_head_options"],
                    output_dim=0,
                )
            },
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
                with self.assertRaisesRegex(ValueError, "dropout probability"):
                    Model(
                        self._direct_config(
                            encoder_options=replace(
                                defaults["encoder_options"],
                                hidden_dim=8,
                                num_layers=2,
                                dropout_probability=probability,
                            )
                        )
                    )

    def test_representative_presets_train_one_tiny_epoch(self):
        for preset in (
            ExperimentPreset.BASELINE,
            ExperimentPreset.TOP1_SWITCH_AUX,
        ):
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                model = Model(cfg)
                datamodule = RandomBertPretrainingDataModule(cfg, batch_size=2)
                tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _config(
        self,
        preset: ExperimentPreset,
        config_overrides: dict | None = None,
    ):
        overrides = {
            **self._small_overrides(),
            **(config_overrides or {}),
        }
        return ExperimentPresets().get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides=overrides,
        )[0]

    def _direct_config(self, **overrides):
        builder_kwargs = expert_linear_builder_kwargs_from_flat(
            self._small_overrides(),
            config,
        )
        builder_kwargs.update(overrides)
        return BertExpertLinearConfigBuilder(**builder_kwargs).build()

    def _small_overrides(self) -> dict:
        defaults = _default_builder_kwargs()
        return {
            "batch_size": 2,
            "sequence_length": 6,
            "encoder_options": replace(
                defaults["encoder_options"],
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
        if isinstance(encoder_config, RecurrentLayerConfig):
            encoder_config = encoder_config.block_config
        return encoder_config.layer_config.layer_model_config

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
