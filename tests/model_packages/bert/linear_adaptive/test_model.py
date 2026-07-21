import importlib
import inspect
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

import models.bert.linear_adaptive.config as config
import models.bert.linear_adaptive.dataset_options as dataset_options
import models.bert.linear_adaptive.runtime_options as runtime_options
from emperor.augmentations.adaptive_parameters import AdaptiveLinearLayerConfig
from emperor.layers import ActivationOptions
from models.bert.linear_adaptive import _config_defaults as config_defaults
from models.bert.linear_adaptive._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.bert.linear_adaptive.config_builder import (
    BertLinearAdaptiveConfigBuilder,
)
from models.bert.linear_adaptive.model import Model
from models.bert.linear_adaptive.presets import (
    Experiment,
    ExperimentPreset,
    ExperimentPresets,
)
from models.catalog import catalog_entry
from models.training_test_utils import (
    RandomBertPretrainingDataModule,
    tiny_cpu_trainer,
)


class TestBertLinearAdaptiveModel(unittest.TestCase):
    def test_public_surface_and_catalog_id(self):
        for module_name in (
            "models.bert.linear_adaptive.config",
            "models.bert.linear_adaptive.presets",
            "models.bert.linear_adaptive.model",
            "models.bert.linear_adaptive.config_builder",
            "models.bert.linear_adaptive.experiment_config",
        ):
            with self.subTest(module_name=module_name):
                module = importlib.import_module(module_name)

                self.assertEqual(module.__name__, module_name)

        self.assertEqual(Experiment()._public_model_id(), "bert/linear_adaptive")
        self.assertIsNotNone(catalog_entry("bert/linear_adaptive"))

        package = importlib.import_module("models.bert.linear_adaptive")
        self.assertEqual(package.__all__, ["Experiment", "ExperimentPreset"])
        self.assertFalse(hasattr(package, "BertLinearAdaptiveConfigBuilder"))
        self.assertFalse(hasattr(package, "ExperimentConfig"))
        self.assertFalse(hasattr(package, "Model"))

    def test_package_avoids_generic_adaptive_config_and_sibling_bert_packages(self):
        package_dir = Path(config.__file__).resolve().parent

        for path in package_dir.glob("*.py"):
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "models.linears.linear_adaptive" + ".config",
                    source,
                )
                self.assertNotIn("models.bert." + "linear.", source)
                self.assertNotIn("models.bert." + "expert_linear", source)

    def test_catalog_has_74_contiguous_unique_presets(self):
        self.assertEqual(len(ExperimentPreset), 74)
        self.assertEqual(
            [preset.value for preset in ExperimentPreset],
            list(range(1, 75)),
        )

    def test_config_builder_uses_supplied_local_adaptive_defaults(self):
        local_hidden_dim = config.HIDDEN_DIM + 11
        adaptive_generator_stack_options = replace(
            self._default_builder_kwargs()["adaptive_generator_stack_options"],
            hidden_dim=local_hidden_dim,
        )

        builder = BertLinearAdaptiveConfigBuilder(
            adaptive_generator_stack_options=adaptive_generator_stack_options,
        )

        self.assertEqual(
            builder.adaptive_generator_stack_options.hidden_dim,
            local_hidden_dim,
        )

    def test_config_builder_has_an_explicit_keyword_only_interface(self):
        parameters = inspect.signature(
            BertLinearAdaptiveConfigBuilder.__init__
        ).parameters

        self.assertNotIn("runtime", parameters)
        self.assertFalse(
            any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        for name, parameter in parameters.items():
            if name != "self":
                with self.subTest(name=name):
                    self.assertIs(
                        parameter.kind,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
        for name in (
            "sequence_length",
            "embedding_options",
            "mlm_head_options",
            "nsp_head_options",
            "attention_projection_layer_controller_options",
            "feed_forward_recurrent_controller_options",
            "attention_hidden_adaptive_weight_options",
            "feed_forward_hidden_adaptive_mask_options",
        ):
            self.assertIn(name, parameters)

    def test_legacy_runtime_modules_and_types_are_removed(self):
        package_dir = Path(config.__file__).resolve().parent
        removed_modules = (
            "runtime_defaults",
            "_adaptive_builder_options",
            "_builder_options",
            "_controller_stack",
            "_linear_builder_options",
            "_transformer_builder_options",
        )

        self.assertFalse(hasattr(runtime_options, "RuntimeOptions"))
        self.assertFalse(hasattr(runtime_options, "VitPatchOptions"))
        self.assertFalse(hasattr(runtime_options, "VitOutputOptions"))
        for module_name in removed_modules:
            with self.subTest(module_name=module_name):
                self.assertFalse((package_dir / f"{module_name}.py").exists())
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(
                        f"models.bert.linear_adaptive.{module_name}"
                    )

    def test_flat_adapter_matches_explicit_grouped_builder_configuration(self):
        flat_options = {
            **self._test_overrides(),
            "weight_option": config.LowRankDynamicWeightConfig,
            "stack_gate_flag": True,
        }
        encoder_options = replace(
            config_defaults.bert_encoder_options(config),
            hidden_dim=flat_options["hidden_dim"],
            num_layers=flat_options["stack_num_layers"],
            dropout_probability=flat_options["stack_dropout_probability"],
        )
        attention_options = replace(
            config_defaults.bert_attention_options(config),
            num_heads=flat_options["attn_num_heads"],
        )
        recurrent_controller_options = replace(
            config_defaults.linears_recurrent_controller_options(
                config,
                recurrent_prefix="RECURRENT",
                gate_stack_prefix="RECURRENT_GATE_STACK",
                halting_stack_prefix="RECURRENT_HALTING_STACK",
            ),
            recurrent_max_steps=flat_options["recurrent_max_steps"],
        )
        layer_controller_options = replace(
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

        self.assertEqual(
            BertLinearAdaptiveConfigBuilder(
                **linear_adaptive_builder_kwargs_from_flat(flat_options, config)
            ).build(),
            BertLinearAdaptiveConfigBuilder(
                batch_size=flat_options["batch_size"],
                sequence_length=flat_options["sequence_length"],
                encoder_options=encoder_options,
                attention_options=attention_options,
                layer_controller_options=layer_controller_options,
                recurrent_controller_options=recurrent_controller_options,
                hidden_adaptive_weight_options=weight_options,
            ).build(),
        )

    def test_unknown_flat_option_reaches_builder_type_error(self):
        builder_kwargs = linear_adaptive_builder_kwargs_from_flat(
            {"unknown_option": 7},
            config,
        )

        self.assertEqual(builder_kwargs["unknown_option"], 7)
        with self.assertRaisesRegex(TypeError, "unknown_option"):
            BertLinearAdaptiveConfigBuilder(**builder_kwargs)

    def test_low_rank_preset_uses_adaptive_projection_and_feed_forward_layers(self):
        cfg = self._config(ExperimentPreset.LOW_RANK_WEIGHT)
        encoder_layer_config = self._encoder_layer_config(cfg)

        projection_stack = encoder_layer_config.attention_config.projection_model_config
        projection_layer_config = projection_stack.layer_config.layer_model_config
        feed_forward_stack = encoder_layer_config.feed_forward_config.stack_config
        feed_forward_layer_config = feed_forward_stack.layer_config.layer_model_config

        self.assertIsInstance(projection_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            projection_layer_config.adaptive_augmentation_config.weight_config
        )
        self.assertIsInstance(feed_forward_layer_config, AdaptiveLinearLayerConfig)
        self.assertIsNotNone(
            feed_forward_layer_config.adaptive_augmentation_config.weight_config
        )

    def test_causal_preset_enables_encoder_and_attention_masking(self):
        encoder_layer_config = self._encoder_layer_config(
            self._config(ExperimentPreset.CAUSAL)
        )

        self.assertTrue(encoder_layer_config.causal_attention_mask_flag)
        self.assertTrue(
            encoder_layer_config.attention_config.causal_attention_mask_flag
        )

    def test_global_weight_override_adapts_attention_and_feed_forward(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
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

    def test_attention_weight_override_is_isolated(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
                "attn_weight_option": config.LowRankDynamicWeightConfig,
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

    def test_feed_forward_weight_override_is_isolated(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
                "ff_weight_option": config.LowRankDynamicWeightConfig,
            },
        )

        self.assertIsNone(
            self._projection_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )
        self.assertIsNotNone(
            self._feed_forward_layer_config(
                cfg
            ).adaptive_augmentation_config.weight_config
        )

    def test_role_false_disables_global_weight(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
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
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
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

    def test_bert_boundaries_remain_non_adaptive(self):
        model = Model(self._config(ExperimentPreset.LOW_RANK_WEIGHT))

        self.assertIsInstance(model.token_embedding, nn.Embedding)
        self.assertIsInstance(model.token_type_embedding, nn.Embedding)
        self.assertIsInstance(model.mlm_dense, nn.Linear)
        self.assertIsInstance(model.mlm_decoder, nn.Linear)
        self.assertIsInstance(model.pooler, nn.Linear)
        self.assertIsInstance(model.nsp_head, nn.Linear)

    def test_stack_controls_keep_adaptive_backend_layers(self):
        defaults = self._default_builder_kwargs()
        cfg = self._config(
            ExperimentPreset.LOW_RANK_WEIGHT,
            {
                **self._test_overrides(),
                "feed_forward_stack_options": replace(
                    defaults["feed_forward_stack_options"],
                    hidden_dim=17,
                ),
                "feed_forward_layer_controller_options": replace(
                    defaults["feed_forward_layer_controller_options"],
                    stack_gate_flag=True,
                ),
                "attention_projection_stack_options": replace(
                    defaults["attention_projection_stack_options"],
                    hidden_dim=19,
                ),
                "attention_projection_layer_controller_options": replace(
                    defaults["attention_projection_layer_controller_options"],
                    stack_gate_flag=True,
                ),
            },
        )
        encoder_layer_config = self._encoder_layer_config(cfg)
        projection_stack = encoder_layer_config.attention_config.projection_model_config
        feed_forward_stack = encoder_layer_config.feed_forward_config.stack_config

        self.assertEqual(projection_stack.hidden_dim, 19)
        self.assertEqual(feed_forward_stack.hidden_dim, 17)
        self.assertIsNotNone(projection_stack.layer_config.gate_config)
        self.assertIsNotNone(feed_forward_stack.layer_config.gate_config)
        self.assertIsInstance(
            projection_stack.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )
        self.assertIsInstance(
            feed_forward_stack.layer_config.layer_model_config,
            AdaptiveLinearLayerConfig,
        )

    def test_configurable_embedding_mlm_and_nsp_boundaries(self):
        cfg = self._config(
            ExperimentPreset.BASELINE,
            {
                **self._test_overrides(),
                "token_type_vocab_size": 4,
                "embedding_layer_norm_flag": False,
                "embedding_dropout_probability": 0.25,
                "mlm_activation": ActivationOptions.RELU,
                "mlm_dense_bias_flag": False,
                "mlm_layer_norm_flag": False,
                "mlm_decoder_bias_flag": False,
                "mlm_decoder_weight_tying_flag": False,
                "nsp_pooler_activation": ActivationOptions.RELU,
                "nsp_pooler_bias_flag": False,
                "nsp_output_dim": 3,
                "nsp_head_bias_flag": False,
            },
        )
        model = Model(cfg)

        self.assertEqual(model.token_type_embedding.num_embeddings, 4)
        self.assertIsInstance(model.embedding_layer_norm, nn.Identity)
        self.assertEqual(model.embedding_dropout.p, 0.25)
        self.assertIsInstance(model.mlm_activation, nn.ReLU)
        self.assertIsNone(model.mlm_dense.bias)
        self.assertIsInstance(model.mlm_layer_norm, nn.Identity)
        self.assertIsNone(model.mlm_decoder_bias)
        self.assertIsNot(model.mlm_decoder.weight, model.token_embedding.weight)
        self.assertIsInstance(model.pooler_activation, nn.ReLU)
        self.assertIsNone(model.pooler.bias)
        self.assertEqual(model.nsp_head.out_features, 3)
        self.assertIsNone(model.nsp_head.bias)

    def test_all_presets_forward_one_batch(self):
        for preset in ExperimentPreset:
            with self.subTest(preset=preset.name):
                cfg = self._config(preset)
                mlm_logits, nsp_logits, auxiliary_loss = Model(cfg)(
                    *self._fake_bert_inputs(cfg)
                )

                self.assertEqual(
                    mlm_logits.shape, (2, cfg.sequence_length, cfg.output_dim)
                )
                self.assertEqual(nsp_logits.shape, (2, 2))
                self.assertEqual(auxiliary_loss.dim(), 0)
                self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_baseline_trains_one_tiny_epoch(self):
        cfg = self._config(ExperimentPreset.BASELINE)
        model = Model(cfg)
        datamodule = RandomBertPretrainingDataModule(cfg, batch_size=2)

        tiny_cpu_trainer().fit(model, datamodule=datamodule)

    def _config(
        self,
        preset: ExperimentPreset,
        config_overrides: dict | None = None,
    ):
        return ExperimentPresets().get_config(
            preset,
            dataset_options.DATASET_OPTIONS_BY_TASK[
                dataset_options.DEFAULT_EXPERIMENT_TASK
            ][0],
            config_overrides=config_overrides or self._test_overrides(),
        )[0]

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

    def _test_overrides(self) -> dict:
        return {
            "batch_size": 2,
            "hidden_dim": 16,
            "sequence_length": 8,
            "stack_num_layers": 2,
            "attn_num_heads": 4,
            "stack_dropout_probability": 0.0,
            "recurrent_max_steps": 2,
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
        if hasattr(encoder_config, "block_config"):
            encoder_config = encoder_config.block_config
        return encoder_config.layer_config.layer_model_config

    def _projection_layer_config(self, cfg):
        return self._encoder_layer_config(
            cfg
        ).attention_config.projection_model_config.layer_config.layer_model_config

    def _feed_forward_layer_config(self, cfg):
        return self._encoder_layer_config(
            cfg
        ).feed_forward_config.stack_config.layer_config.layer_model_config


if __name__ == "__main__":
    unittest.main()
