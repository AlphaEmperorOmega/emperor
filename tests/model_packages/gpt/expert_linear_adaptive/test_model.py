import importlib
import inspect
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

import models.gpt.expert_linear_adaptive.config as config
import models.gpt.expert_linear_adaptive.dataset_options as dataset_options
from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.transformer import TransformerDecoderLayer, TransformerDecoderLayerState
from models.catalog import MODEL_CATALOG, catalog_entry
from models.gpt.expert_linear_adaptive import (
    Experiment,
    ExperimentConfig,
    ExperimentPreset,
    ExperimentPresets,
    GptExpertLinearAdaptiveConfigBuilder,
    GptLmHeadOptions,
    Model,
)
from models.gpt.expert_linear_adaptive.runtime_defaults import (
    expert_linear_adaptive_builder_kwargs_from_flat,
)
from models.training_test_utils import (
    RandomLanguageModelDataModule,
    tiny_cpu_trainer,
)

_MIXTURE_ATTENTION_TYPE = MixtureOfAttentionHeadsConfig().registry_owner()
_MIXTURE_OF_EXPERTS_TYPE = MixtureOfExpertsModelConfig().registry_owner()
_MIXTURE_OF_EXPERTS_LAYER_TYPE = MixtureOfExpertsConfig().registry_owner()
_SELF_ATTENTION_TYPE = SelfAttentionConfig().registry_owner()


class TestGptExpertLinearAdaptiveModel(unittest.TestCase):
    backend_module_name = "AdaptiveLinearLayer"

    def config(self, **overrides):
        return GptExpertLinearAdaptiveConfigBuilder(
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
            MODEL_CATALOG["gpt/expert_linear_adaptive"].module_path,
            "models.gpt.expert_linear_adaptive",
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
        self.assertIn(
            "MixtureOfExperts",
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
            GptExpertLinearAdaptiveConfigBuilder(
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
            "models.gpt.expert_linear_adaptive.config",
            "models.gpt.expert_linear_adaptive.presets",
            "models.gpt.expert_linear_adaptive.model",
            "models.gpt.expert_linear_adaptive.config_builder",
            "models.gpt.expert_linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                self.assertEqual(
                    importlib.import_module(module_name).__name__,
                    module_name,
                )
        self.assertEqual(
            Experiment()._public_model_id(),
            "gpt/expert_linear_adaptive",
        )
        self.assertIsNotNone(catalog_entry("gpt/expert_linear_adaptive"))

    def test_package_avoids_bert_and_gpt_sibling_construction_imports(self):
        package_dir = Path(config.__file__).resolve().parent
        blocked = (
            "models.bert.",
            "models.gpt.linear.",
            "models.gpt.linear_adaptive.",
            "models.gpt.expert_linear.",
        )
        for path in package_dir.glob("*.py"):
            if path.name == "test_model.py":
                continue
            source = path.read_text(encoding="utf-8")
            for import_path in blocked:
                with self.subTest(path=path.name, import_path=import_path):
                    self.assertNotIn(import_path, source)

    def test_builder_removes_attention_and_causal_switches(self):
        parameters = inspect.signature(GptExpertLinearAdaptiveConfigBuilder).parameters
        self.assertNotIn("expert_attention_flag", parameters)
        self.assertNotIn("causal_attention_mask_flag", parameters)
        self.assertIn("expert_attention_use_kv_expert_models_flag", parameters)
        self.assertIn("lm_head_options", parameters)
        self.assertIn("embedding_options", parameters)
        self.assertFalse(hasattr(config, "EXPERT_ATTENTION_FLAG"))
        self.assertFalse(hasattr(config, "CAUSAL_ATTENTION_MASK_FLAG"))
        with self.assertRaises(TypeError):
            GptExpertLinearAdaptiveConfigBuilder(expert_attention_flag=False)
        with self.assertRaises(TypeError):
            GptExpertLinearAdaptiveConfigBuilder(causal_attention_mask_flag=False)

    def test_low_rank_preset_uses_adaptive_expert_layers(self):
        cfg = self._preset_config(ExperimentPreset.LOW_RANK_EXPERT_WEIGHT)
        mixture = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        expert_core = mixture.stack_config.layer_config.layer_model_config
        expert_layer = expert_core.expert_model_config.layer_config.layer_model_config
        self.assertIsInstance(mixture, MixtureOfExpertsModelConfig)
        self.assertIsInstance(expert_layer, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(expert_layer.adaptive_augmentation_config.weight_config)

    def test_expert_attention_uses_adaptive_expert_internals(self):
        cfg = self._preset_config(ExperimentPreset.BASELINE)
        attention = self._decoder_layer_config(cfg).self_attention_config
        expert_stack = attention.experts_config.expert_model_config
        expert_layer = expert_stack.layer_config.layer_model_config
        self.assertIsInstance(attention, MixtureOfAttentionHeadsConfig)
        self.assertIsInstance(expert_layer, AdaptiveLinearLayerConfig)

    def test_ff_controls_apply_to_outer_slot_and_preserve_adaptive_experts(self):
        defaults = self._default_builder_kwargs()
        cfg = self._preset_config(
            ExperimentPreset.LOW_RANK_EXPERT_WEIGHT,
            {
                "feed_forward_stack_options": replace(
                    defaults["feed_forward_stack_options"],
                    hidden_dim=17,
                ),
                "feed_forward_layer_controller_options": replace(
                    defaults["feed_forward_layer_controller_options"],
                    stack_gate_flag=True,
                ),
                "expert_stack_options": replace(
                    defaults["expert_stack_options"],
                    hidden_dim=11,
                ),
            },
        )
        mixture = self._decoder_layer_config(cfg).feed_forward_config.stack_config
        outer = mixture.stack_config
        expert_stack = outer.layer_config.layer_model_config.expert_model_config
        expert_layer = expert_stack.layer_config.layer_model_config
        self.assertEqual(outer.hidden_dim, 17)
        self.assertIsNotNone(outer.layer_config.gate_config)
        self.assertEqual(expert_stack.hidden_dim, 11)
        self.assertIsInstance(expert_layer, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(expert_layer.adaptive_augmentation_config.weight_config)

    def test_attention_controls_preserve_adaptive_expert_attention(self):
        defaults = self._default_builder_kwargs()
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "hidden_adaptive_weight_options": replace(
                    defaults["hidden_adaptive_weight_options"],
                    option_flag=True,
                    option=config.LowRankDynamicWeightConfig,
                ),
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
        expert_stack = attention.experts_config.expert_model_config
        expert_layer = expert_stack.layer_config.layer_model_config
        self.assertEqual(projection.hidden_dim, 17)
        self.assertIsNotNone(projection.layer_config.gate_config)
        self.assertEqual(expert_stack.hidden_dim, 11)
        self.assertIsNone(expert_stack.layer_config.gate_config)
        self.assertIsInstance(expert_layer, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(expert_layer.adaptive_augmentation_config.weight_config)

    def test_every_preset_forwards_adaptive_moe_and_mixture_attention(self):
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
                self.assertTrue(
                    any(type(m).__name__ == "AdaptiveLinearLayer" for m in modules)
                )
                self.assertFalse(
                    any(isinstance(m, _SELF_ATTENTION_TYPE) for m in modules)
                )
                self.assertEqual(
                    tuple(logits.shape),
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(tuple(auxiliary_loss.shape), ())
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(auxiliary_loss))

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
                (logits.square().mean() + auxiliary_loss).backward()
                self._assert_nonzero_parameter_gradients(
                    model.token_embedding,
                    "token embedding",
                )
                projector = attention.projector
                if use_kv_experts:
                    for role, expert_model in {
                        "key": projector.key_model,
                        "value": projector.value_model,
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

    def test_routed_auxiliary_loss_is_added_to_cross_entropy(self):
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

    def test_attention_mask_becomes_padding_and_causal_decoder_masks(self):
        cfg = self._preset_config(ExperimentPreset.BASELINE)
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
        attention_mask[:, -1] = 0
        model(input_ids, attention_mask)
        self.assertIsInstance(spy.state, TransformerDecoderLayerState)
        torch.testing.assert_close(
            spy.state.target_key_padding_mask,
            attention_mask == 0,
        )
        expected = torch.triu(
            torch.ones(
                cfg.sequence_length,
                cfg.sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        torch.testing.assert_close(spy.state.target_attention_mask, expected)

    def test_boundary_options_are_plain_and_fully_configurable(self):
        defaults = self._default_builder_kwargs()
        cfg = self._direct_config(
            embedding_options=replace(
                defaults["embedding_options"],
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
        self.assertIsInstance(model.token_embedding, nn.Embedding)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertIsInstance(model.embedding_layer_norm, nn.Identity)
        self.assertEqual(model.embedding_dropout.p, 0.25)
        self.assertIsNot(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNotNone(model.lm_head.bias)

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
                overrides = self._small_flat_overrides()
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
            ExperimentPreset.LOW_RANK_EXPERT_WEIGHT,
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
                **self._small_flat_overrides(),
                **(overrides or {}),
            },
        )[0]

    def _direct_config(self, **overrides):
        kwargs = expert_linear_adaptive_builder_kwargs_from_flat(
            self._small_flat_overrides(),
            config,
        )
        kwargs.update(overrides)
        return GptExpertLinearAdaptiveConfigBuilder(**kwargs).build()

    def _small_flat_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "input_dim": 32,
            "output_dim": 32,
            "hidden_dim": 8,
            "sequence_length": 6,
            "stack_num_layers": 2,
            "attn_num_heads": 2,
            "stack_dropout_probability": 0.0,
            "num_experts": 4,
            "recurrent_max_steps": 2,
        }

    def _default_builder_kwargs(self) -> dict:
        return expert_linear_adaptive_builder_kwargs_from_flat({}, config)

    def _input_ids(self, cfg) -> torch.Tensor:
        return torch.randint(0, cfg.input_dim, (2, cfg.sequence_length))

    def _decoder_layer_config(self, cfg):
        decoder = cfg.experiment_config.decoder_config
        decoder = getattr(decoder, "block_config", decoder)
        return decoder.layer_config.layer_model_config

    def _assert_nonzero_parameter_gradients(self, model, role: str) -> None:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients, f"{role} had no gradients")
        self.assertTrue(
            any(gradient.abs().sum().item() > 0.0 for gradient in gradients),
            f"{role} gradients were all zero",
        )


if __name__ == "__main__":
    unittest.main()
