import importlib
import inspect
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

import models.gpt.linear_adaptive.config as config
import models.gpt.linear_adaptive.dataset_options as dataset_options
import models.gpt.linear_adaptive.runtime_options as runtime_options
from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.experiments.language_model import LanguageModelExperiment
from emperor.layers import ActivationOptions, LayerNormPositionOptions
from emperor.transformer import (
    TransformerDecoderLayerConfig,
    TransformerDecoderLayerState,
)
from models.catalog import model_package
from models.gpt.linear_adaptive import _config_defaults as config_defaults
from models.gpt.linear_adaptive._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.gpt.linear_adaptive.config_builder import GptLinearAdaptiveConfigBuilder
from models.gpt.linear_adaptive.experiment_config import ExperimentConfig
from models.gpt.linear_adaptive.model import Model
from models.gpt.linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.gpt.linear_adaptive.runtime_options import RuntimeOptions
from models.training_test_utils import (
    RandomLanguageModelDataModule,
    tiny_cpu_trainer,
)

_TRANSFORMER_DECODER_LAYER_TYPE = TransformerDecoderLayerConfig().registry_owner()


class TestGptLinearAdaptiveModel(unittest.TestCase):
    backend_module_name = "AdaptiveLinearLayer"

    def test_runtime_defaults_describe_a_gpt2_decoder_block(self):
        self.assertEqual(
            config.LAYER_NORM_POSITION,
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

    def config(self, **overrides):
        runtime = model_package("gpt/linear_adaptive").bind_runtime_defaults(
            {
                "input_dim": 16,
                "output_dim": 16,
                "sequence_length": 6,
                **overrides,
            }
        )
        return GptLinearAdaptiveConfigBuilder(runtime=runtime).build()

    def test_public_imports_and_catalog_identity(self):
        self.assertTrue(issubclass(Model, LanguageModelExperiment))
        self.assertIsNotNone(Experiment)
        self.assertIsNotNone(ExperimentConfig)
        self.assertIsNotNone(ExperimentPresets)
        self.assertEqual(
            model_package("gpt/linear_adaptive").catalog_key,
            "gpt/linear_adaptive",
        )

    def test_presets_are_contiguous_buildable_and_always_causal(self):
        presets = model_package("gpt/linear_adaptive").presets
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
            if isinstance(module, _TRANSFORMER_DECODER_LAYER_TYPE)
        ]
        self.assertTrue(decoder_layers)
        self.assertTrue(
            all(layer.cross_attention_model is None for layer in decoder_layers)
        )

    def test_lm_head_options_allow_untied_biased_projection(self):
        cfg = self.config(
            lm_head_weight_tying_flag=False,
            lm_head_bias_flag=True,
        )
        model = Model(cfg)
        self.assertIsNot(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNotNone(model.lm_head.bias)

    def test_tied_head_rejects_unequal_vocabularies(self):
        with self.assertRaises(ValueError):
            self.config(
                input_dim=15,
                output_dim=16,
                sequence_length=4,
            )

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

    def test_forward_leaves_causality_to_attention(self):
        model = Model(self.config())

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
        model(torch.tensor([[1, 2, 3, 4]]))

        self.assertIsInstance(spy.state, TransformerDecoderLayerState)
        self.assertIsNone(spy.state.target_attention_mask)

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
            "models.gpt.linear_adaptive.config",
            "models.gpt.linear_adaptive.presets",
            "models.gpt.linear_adaptive.model",
            "models.gpt.linear_adaptive.config_builder",
            "models.gpt.linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                self.assertEqual(
                    importlib.import_module(module_name).__name__,
                    module_name,
                )
        experiment = Experiment(model_package=model_package("gpt/linear_adaptive"))
        self.assertEqual(
            experiment.model_package.identity.catalog_key,
            "gpt/linear_adaptive",
        )
        self.assertIsNotNone(model_package("gpt/linear_adaptive"))

    def test_package_avoids_bert_and_gpt_sibling_construction_imports(self):
        package_dir = Path(config.__file__).resolve().parent
        blocked = (
            "models.bert.",
            "models.gpt.linear.",
            "models.gpt.expert_linear",
            "models.linears.linear_adaptive.config",
        )
        for path in package_dir.glob("*.py"):
            if path.name == "test_model.py":
                continue
            source = path.read_text(encoding="utf-8")
            for import_path in blocked:
                with self.subTest(path=path.name, import_path=import_path):
                    self.assertNotIn(import_path, source)

    def test_builder_accepts_only_typed_runtime_options(self):
        parameters = inspect.signature(
            GptLinearAdaptiveConfigBuilder.__init__
        ).parameters
        self.assertEqual(tuple(parameters), ("self", "runtime"))
        self.assertFalse(
            any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        self.assertIs(
            parameters["runtime"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for removed_name in (
            "encoder_options",
            "causal_attention_mask_flag",
            "token_type_vocab_size",
            "mlm_head_options",
            "nsp_head_options",
        ):
            self.assertNotIn(removed_name, parameters)

    def test_removed_legacy_runtime_modules_stay_removed(self):
        package_dir = Path(config.__file__).resolve().parent
        removed_modules = (
            "_adaptive_builder_options",
            "_builder_options",
            "_controller_stack",
            "_linear_builder_options",
            "_transformer_builder_options",
        )
        self.assertIs(runtime_options.RuntimeOptions, RuntimeOptions)
        self.assertEqual(
            RuntimeOptions.__module__,
            "models.gpt.linear_adaptive.runtime_options",
        )
        self.assertTrue((package_dir / "runtime_defaults.py").is_file())
        for module_name in removed_modules:
            with self.subTest(module_name=module_name):
                self.assertFalse((package_dir / f"{module_name}.py").exists())
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"models.gpt.linear_adaptive.{module_name}")

    def test_flat_runtime_defaults_bind_to_typed_runtime_options(self):
        flat_options = {
            **self._small_overrides(),
            "weight_option": config.LowRankDynamicWeightConfig,
            "stack_gate_flag": True,
        }
        decoder_options = replace(
            config_defaults.gpt_decoder_options(config),
            hidden_dim=flat_options["hidden_dim"],
            num_layers=flat_options["stack_num_layers"],
            dropout_probability=flat_options["stack_dropout_probability"],
        )
        attention_options = replace(
            config_defaults.gpt_attention_options(config),
            num_heads=flat_options["attn_num_heads"],
        )
        recurrent_options = replace(
            config_defaults.linears_recurrent_controller_options(
                config,
                recurrent_prefix="RECURRENT",
                gate_stack_prefix="RECURRENT_GATE_STACK",
                halting_stack_prefix="RECURRENT_HALTING_STACK",
            ),
            recurrent_max_steps=flat_options["recurrent_max_steps"],
        )
        layer_options = replace(
            config_defaults.linears_layer_controller_options(
                config,
                gate_prefix="GATE",
                gate_stack_prefix="GATE_STACK",
                halting_prefix="HALTING",
                halting_stack_prefix="HALTING_STACK",
            ),
            stack_gate_flag=True,
        )
        weight_options = replace(
            config_defaults.hidden_adaptive_weight_options(config),
            option_flag=True,
            option=config.LowRankDynamicWeightConfig,
        )
        adapted = linear_adaptive_builder_kwargs_from_flat(flat_options, config)
        self.assertEqual(adapted["decoder_options"], decoder_options)
        self.assertEqual(adapted["attention_options"], attention_options)
        self.assertEqual(adapted["layer_controller_options"], layer_options)
        self.assertEqual(adapted["recurrent_controller_options"], recurrent_options)
        self.assertEqual(adapted["hidden_adaptive_weight_options"], weight_options)

        runtime = model_package("gpt/linear_adaptive").bind_runtime_defaults(
            flat_options
        )
        self.assertEqual(runtime, RuntimeOptions(adapted))
        GptLinearAdaptiveConfigBuilder(runtime=runtime).build()

    def test_unknown_runtime_default_is_rejected_at_package_boundary(self):
        with self.assertRaisesRegex(ValueError, "unknown_option"):
            model_package("gpt/linear_adaptive").bind_runtime_defaults(
                {"unknown_option": 7}
            )

    def test_low_rank_preset_adapts_projection_and_feed_forward_layers(self):
        cfg = self._preset_config(ExperimentPreset.LOW_RANK_WEIGHT)
        projection = self._projection_layer_config(cfg)
        feed_forward = self._feed_forward_layer_config(cfg)
        for role, layer_config in (
            ("attention", projection),
            ("feed_forward", feed_forward),
        ):
            with self.subTest(role=role):
                self.assertIsInstance(layer_config, AdaptiveLinearLayerConfig)
                self.assertIsNotNone(
                    layer_config.adaptive_augmentation_config.weight_config
                )

    def test_global_adaptive_weight_reaches_both_decoder_roles(self):
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
            },
        )
        self.assertIsNotNone(
            self._projection_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )
        self.assertIsNotNone(
            self._feed_forward_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )

    def test_role_specific_adaptive_weight_overrides_are_isolated(self):
        cases = (
            ("attention", "attn_weight_option", True, False),
            ("feed_forward", "ff_weight_option", False, True),
        )
        for role, option_name, attention_enabled, ff_enabled in cases:
            with self.subTest(role=role):
                cfg = self._preset_config(
                    ExperimentPreset.BASELINE,
                    {option_name: config.LowRankDynamicWeightConfig},
                )
                projection_weight = self._projection_layer_config(
                    cfg
                ).adaptive_augmentation_config.weight_config
                feed_forward_weight = self._feed_forward_layer_config(
                    cfg
                ).adaptive_augmentation_config.weight_config
                self.assertEqual(projection_weight is not None, attention_enabled)
                self.assertEqual(feed_forward_weight is not None, ff_enabled)

    def test_role_false_disables_inherited_global_weight(self):
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
                "ff_weight_option_flag": False,
            },
        )
        self.assertIsNotNone(
            self._projection_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )
        self.assertIsNone(
            self._feed_forward_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )

    def test_role_generator_stack_dimensions_are_independent(self):
        cfg = self._preset_config(
            ExperimentPreset.BASELINE,
            {
                "weight_option_flag": True,
                "weight_option": config.LowRankDynamicWeightConfig,
                "attn_weight_generator_stack_independent_flag": True,
                "attn_weight_generator_stack_hidden_dim": 23,
                "ff_weight_generator_stack_independent_flag": True,
                "ff_weight_generator_stack_hidden_dim": 37,
            },
        )
        projection_weight = self._projection_layer_config(
            cfg
        ).adaptive_augmentation_config.weight_config
        feed_forward_weight = self._feed_forward_layer_config(
            cfg
        ).adaptive_augmentation_config.weight_config
        self.assertEqual(projection_weight.model_config.hidden_dim, 23)
        self.assertEqual(feed_forward_weight.model_config.hidden_dim, 37)

    def test_boundaries_remain_plain_and_configurable(self):
        cfg = self._preset_config(
            ExperimentPreset.LOW_RANK_WEIGHT,
            {
                "embedding_layer_norm_flag": False,
                "embedding_dropout_probability": 0.25,
                "lm_head_weight_tying_flag": False,
                "lm_head_bias_flag": True,
            },
        )
        model = Model(cfg)
        self.assertIsInstance(model.token_embedding, nn.Embedding)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertIsInstance(model.embedding_layer_norm, nn.Identity)
        self.assertEqual(model.embedding_dropout.p, 0.25)
        self.assertIsNot(model.lm_head.weight, model.token_embedding.weight)
        self.assertIsNotNone(model.lm_head.bias)

    def test_stack_controls_preserve_adaptive_backend_layers(self):
        cfg = self._preset_config(
            ExperimentPreset.LOW_RANK_WEIGHT,
            {
                "ff_stack_hidden_dim": 17,
                "ff_stack_gate_flag": True,
                "attn_stack_hidden_dim": 19,
                "attn_stack_gate_flag": True,
            },
        )
        layer = self._decoder_layer_config(cfg)
        projection = layer.self_attention_config.projection_model_config
        feed_forward = layer.feed_forward_config.stack_config
        self.assertEqual(projection.hidden_dim, 19)
        self.assertEqual(feed_forward.hidden_dim, 17)
        self.assertIsNotNone(projection.layer_config.gate_config)
        self.assertIsNotNone(feed_forward.layer_config.gate_config)
        self.assertIsInstance(
            projection.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            feed_forward.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )

    def test_every_preset_forwards_a_finite_causal_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._preset_config(preset)
                logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
                self.assertEqual(
                    tuple(logits.shape),
                    (2, cfg.sequence_length, cfg.output_dim),
                )
                self.assertEqual(tuple(auxiliary_loss.shape), ())
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(auxiliary_loss))
                layer = self._decoder_layer_config(cfg)
                self.assertTrue(layer.self_attention_config.causal_attention_mask_flag)
                self.assertIsNone(layer.cross_attention_config)

    def test_baseline_forwards_both_language_model_datasets(self):
        datasets = dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]
        for dataset in datasets:
            with self.subTest(dataset=dataset.__name__):
                overrides = self._small_overrides()
                overrides.pop("input_dim")
                overrides.pop("output_dim")
                cfg = model_package("gpt/linear_adaptive").presets.get_config(
                    ExperimentPreset.BASELINE,
                    dataset,
                    config_overrides=overrides,
                )[0]
                logits, auxiliary_loss = Model(cfg)(self._input_ids(cfg))
                self.assertEqual(logits.shape[-1], dataset.num_classes)
                self.assertEqual(tuple(auxiliary_loss.shape), ())

    def test_baseline_trains_one_tiny_epoch(self):
        cfg = self._preset_config(ExperimentPreset.BASELINE)
        tiny_cpu_trainer().fit(
            Model(cfg),
            datamodule=RandomLanguageModelDataModule(
                cfg,
                batch_size=2,
                num_batches=1,
            ),
        )

    def test_dimension_and_embedding_dropout_validation_matrix(self):
        cases = {
            "input_dim": {"input_dim": 0},
            "output_dim": {
                "output_dim": 0,
                "lm_head_weight_tying_flag": False,
            },
            "sequence_length": {"sequence_length": 0},
        }
        for field, overrides in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self.config(**overrides)
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "dropout_probability"):
                    self.config(embedding_dropout_probability=probability)

    def _preset_config(
        self,
        preset: ExperimentPreset,
        overrides: dict | None = None,
    ):
        return model_package("gpt/linear_adaptive").presets.get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides={
                **self._small_overrides(),
                **(overrides or {}),
            },
        )[0]

    def _small_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "input_dim": 32,
            "output_dim": 32,
            "hidden_dim": 16,
            "sequence_length": 6,
            "stack_num_layers": 2,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "recurrent_max_steps": 2,
        }

    def _default_builder_kwargs(self) -> dict:
        return {
            "adaptive_generator_stack_options": (
                config_defaults.adaptive_generator_stack_options(config)
            ),
            "feed_forward_stack_options": (
                config_defaults.linears_submodule_stack_options(
                    config,
                    "FF_STACK",
                    num_layers_key="FF_NUM_LAYERS",
                    bias_key="FF_BIAS_FLAG",
                )
            ),
            "feed_forward_layer_controller_options": (
                config_defaults.linears_layer_controller_options(
                    config,
                    gate_prefix="FF_GATE",
                    gate_stack_prefix="FF_GATE_STACK",
                    halting_prefix="FF_HALTING",
                    halting_stack_prefix="FF_HALTING_STACK",
                )
            ),
            "attention_projection_stack_options": (
                config_defaults.linears_submodule_stack_options(
                    config,
                    "ATTN_STACK",
                    num_layers_key="ATTN_NUM_LAYERS",
                    bias_key="ATTN_BIAS_FLAG",
                )
            ),
            "attention_projection_layer_controller_options": (
                config_defaults.linears_layer_controller_options(
                    config,
                    gate_prefix="ATTN_GATE",
                    gate_stack_prefix="ATTN_GATE_STACK",
                    halting_prefix="ATTN_HALTING",
                    halting_stack_prefix="ATTN_HALTING_STACK",
                )
            ),
        }

    def _input_ids(self, cfg) -> torch.Tensor:
        return torch.randint(
            0,
            cfg.input_dim,
            (2, cfg.sequence_length),
        )

    def _decoder_layer_config(self, cfg):
        decoder = cfg.experiment_config.decoder_config
        decoder = getattr(decoder, "block_config", decoder)
        return decoder.layer_config.layer_model_config

    def _projection_layer_config(self, cfg):
        return self._decoder_layer_config(
            cfg
        ).self_attention_config.projection_model_config.layer_config.layer_model_config

    def _feed_forward_layer_config(self, cfg):
        return self._decoder_layer_config(
            cfg
        ).feed_forward_config.stack_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()
