import argparse
import unittest

import models.transformer_encoder.bert_linear.config as bert_linear_config
import models.transformer_encoder.vit_linear.config as vit_linear_config
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    LowRankDynamicWeightConfig,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.options import ActivationOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import GridSearch
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from lightning.pytorch.callbacks import EarlyStopping
from models.config_overrides import iter_supported_config_keys
from models.linears.linear_adaptive import ExperimentOptions
from models.linears.linear_adaptive.presets import Experiment, ExperimentPresets
from models.parser import get_experiment_parser, resolve_experiment_mode


class ExperimentConfigOverrideTestCase:
    def make_parser(self):
        return get_experiment_parser(
            ExperimentOptions.names(),
            "models.linears.linear_adaptive",
        )

    def resolve_args(self, args):
        return resolve_experiment_mode(args, ExperimentOptions)


class TestExperimentConfigOverrideParsing(
    ExperimentConfigOverrideTestCase,
    unittest.TestCase,
):
    def test_resolve_preset_returns_primary_without_selected_options(self):
        args = self.make_parser().parse_args(["--preset", "single-model-weight"])

        config_option, selected_options, search_mode, *_ = self.resolve_args(args)

        self.assertIs(config_option, ExperimentOptions.SINGLE_MODEL_WEIGHT)
        self.assertIsNone(selected_options)
        self.assertIsNone(search_mode)

    def test_resolve_presets_returns_primary_and_ordered_selected_options(self):
        args = self.make_parser().parse_args(
            ["--presets", "single-model-weight", "dual-model-weight"]
        )

        config_option, selected_options, search_mode, *_ = self.resolve_args(args)

        self.assertIs(config_option, ExperimentOptions.SINGLE_MODEL_WEIGHT)
        self.assertEqual(
            selected_options,
            [
                ExperimentOptions.SINGLE_MODEL_WEIGHT,
                ExperimentOptions.DUAL_MODEL_WEIGHT,
            ],
        )
        self.assertIsNone(search_mode)

    def test_resolve_all_presets_preserves_all_options_mode(self):
        args = self.make_parser().parse_args(["--all-presets"])

        config_option, selected_options, search_mode, *_ = self.resolve_args(args)

        self.assertIsNone(config_option)
        self.assertIsNone(selected_options)
        self.assertIsNone(search_mode)

    def test_named_config_flags_parse_supported_value_types(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--num-epochs",
                "30",
                "--callback-early-stopping-patience",
                "0",
                "--learning-rate",
                "1e-4",
                "--hidden-dim",
                "128",
                "--stack-activation",
                "GELU",
                "--gate-option",
                "ADDITION",
                "--recurrent-gate-option",
                "MULTIPLIER",
                "--weight-option",
                "LowRankDynamicWeightConfig",
                "--input-layer-model-option",
                "AdaptiveLinearLayerConfig",
                "--output-layer-model-option",
                "None",
                "--input-layer-weight-option",
                "LowRankDynamicWeightConfig",
            ]
        )

        _, _, _, _, overrides, search_overrides = self.resolve_args(args)

        self.assertEqual(overrides["num_epochs"], 30)
        self.assertEqual(overrides["callback_early_stopping_patience"], 0)
        self.assertEqual(overrides["learning_rate"], 1e-4)
        self.assertEqual(overrides["hidden_dim"], 128)
        self.assertIs(overrides["stack_activation"], ActivationOptions.GELU)
        self.assertIs(overrides["gate_option"], LayerGateOptions.ADDITION)
        self.assertIs(
            overrides["recurrent_gate_option"],
            LayerGateOptions.MULTIPLIER,
        )
        self.assertIs(overrides["weight_option"], LowRankDynamicWeightConfig)
        self.assertIs(
            overrides["input_layer_model_option"],
            AdaptiveLinearLayerConfig,
        )
        self.assertIsNone(overrides["output_layer_model_option"])
        self.assertIs(
            overrides["input_layer_weight_option"],
            LowRankDynamicWeightConfig,
        )
        self.assertEqual(search_overrides, {})

    def test_named_config_flags_parse_bool_and_string_values(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--bias-flag",
                "false",
                "--trainer-accelerator",
                "mps",
            ]
        )

        _, _, _, _, overrides, search_overrides = self.resolve_args(args)

        self.assertIs(overrides["bias_flag"], False)
        self.assertEqual(overrides["trainer_accelerator"], "mps")
        self.assertEqual(search_overrides, {})

    def test_config_marker_only_groups_overrides(self):
        plain_args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--num-epochs",
                "30",
                "--bias-flag",
                "false",
            ]
        )
        grouped_args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--config",
                "--num-epochs",
                "30",
                "--bias-flag",
                "false",
            ]
        )

        _, _, _, _, plain_overrides, plain_search_overrides = self.resolve_args(
            plain_args
        )
        _, _, _, _, grouped_overrides, grouped_search_overrides = self.resolve_args(
            grouped_args
        )

        self.assertEqual(grouped_overrides, plain_overrides)
        self.assertEqual(grouped_search_overrides, plain_search_overrides)

    def test_search_set_parses_lists_using_config_value_types(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "hidden-dim=64,128",
                "--search-set",
                "stack_activation=RELU,GELU",
                "--search-set",
                "gate-option=MULTIPLIER,ADDITION",
                "--search-set",
                "recurrent_gate_option=MULTIPLIER,ADDITION",
                "--search-set",
                "weight_option=None,LowRankDynamicWeightConfig",
                "--search-set",
                "input_layer_model_option=None,AdaptiveLinearLayerConfig",
                "--search-set",
                "input_layer_weight_option=None,LowRankDynamicWeightConfig",
            ]
        )

        _, _, search_mode, _, _, search_overrides = self.resolve_args(args)

        self.assertIsInstance(search_mode, GridSearch)
        self.assertEqual(search_overrides["hidden_dim"], [64, 128])
        self.assertEqual(
            search_overrides["stack_activation"],
            [ActivationOptions.RELU, ActivationOptions.GELU],
        )
        self.assertEqual(
            search_overrides["gate_option"],
            [
                LayerGateOptions.MULTIPLIER,
                LayerGateOptions.ADDITION,
            ],
        )
        self.assertEqual(
            search_overrides["recurrent_gate_option"],
            [
                LayerGateOptions.MULTIPLIER,
                LayerGateOptions.ADDITION,
            ],
        )
        self.assertEqual(
            search_overrides["weight_option"],
            [None, LowRankDynamicWeightConfig],
        )
        self.assertEqual(
            search_overrides["input_layer_model_option"],
            [None, AdaptiveLinearLayerConfig],
        )
        self.assertEqual(
            search_overrides["input_layer_weight_option"],
            [None, LowRankDynamicWeightConfig],
        )

    def test_search_set_requires_search_mode(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--search-set",
                "hidden_dim=64,128",
            ]
        )

        with self.assertRaises(ValueError):
            self.resolve_args(args)

    def test_malformed_search_set_raises_argument_error(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "hidden_dim",
            ]
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            self.resolve_args(args)

    def test_unknown_search_set_key_raises_argument_error(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--grid-search",
                "--search-set",
                "does_not_exist=1,2",
            ]
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            self.resolve_args(args)

    def test_fixed_override_and_search_axis_conflict_raises(self):
        args = self.make_parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--hidden-dim",
                "128",
                "--grid-search",
                "--search-set",
                "hidden_dim=64,128",
            ]
        )

        with self.assertRaises(ValueError):
            self.resolve_args(args)

    def test_transformer_configs_hide_builder_unsupported_override_keys(self):
        bert_keys = set(iter_supported_config_keys(bert_linear_config))
        vit_keys = set(iter_supported_config_keys(vit_linear_config))

        self.assertNotIn("GATE_FLAG", bert_keys)
        self.assertNotIn("HALTING_FLAG", bert_keys)
        self.assertNotIn("RECURRENT_FLAG", bert_keys)
        self.assertNotIn("BERT_PRETRAINING_TARGET_VOCAB_SIZE", bert_keys)
        self.assertNotIn("SEQUENCE_LENGTH", vit_keys)
        self.assertNotIn("BIAS_FLAG", vit_keys)


class TestExperimentConfigOverrideApplication(unittest.TestCase):
    def test_cli_override_wins_over_selected_preset_default(self):
        cfg = ExperimentPresets().get_config(
            ExperimentOptions.SINGLE_MODEL_WEIGHT,
            Mnist,
            config_overrides={
                "hidden_dim": 64,
                "weight_option": LowRankDynamicWeightConfig,
            },
        )[0]
        adaptive_config = (
            cfg.experiment_config.model_config.layer_config.layer_model_config
            .adaptive_augmentation_config
        )

        self.assertEqual(cfg.hidden_dim, 64)
        self.assertIsInstance(
            adaptive_config.weight_config,
            LowRankDynamicWeightConfig,
        )

    def test_search_overrides_create_expected_configs(self):
        configs = ExperimentPresets().get_config(
            ExperimentOptions.SINGLE_MODEL_WEIGHT,
            Mnist,
            search_mode=GridSearch(),
            search_overrides={
                "hidden_dim": [64, 128],
                "stack_num_layers": [2, 4],
            },
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            sorted({cfg.hidden_dim for cfg in configs}),
            [64, 128],
        )
        self.assertEqual(
            sorted({cfg.experiment_config.model_config.num_layers for cfg in configs}),
            [2, 4],
        )

    def test_trainer_overrides_disable_early_stopping_without_static_monitors(self):
        experiment = Experiment(ExperimentOptions.SINGLE_MODEL_WEIGHT)

        trainer_config = experiment._load_trainer_config(
            {
                "callback_early_stopping_patience": 0,
                "trainer_devices": 1,
            }
        )
        callback_types = {type(callback) for callback in trainer_config["callbacks"]}

        self.assertNotIn(EarlyStopping, callback_types)
        self.assertNotIn(AdaptiveParameterMonitorCallback, callback_types)
        self.assertEqual(trainer_config["trainer_args"]["devices"], 1)


if __name__ == "__main__":
    unittest.main()
