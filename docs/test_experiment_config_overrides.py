import argparse
import unittest

from emperor.augmentations.adaptive_parameters.core.weight import (
    LowRankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.base.options import ActivationOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import GridSearch
from lightning.pytorch.callbacks import EarlyStopping
from models.linear_adaptive import ExperimentOptions
from models.linear_adaptive.presets import Experiment, ExperimentPresets
from models.parser import get_experiment_parser, resolve_experiment_mode


class ExperimentConfigOverrideTestCase:
    def make_parser(self):
        return get_experiment_parser(
            ExperimentOptions.names(),
            "models.linear_adaptive",
        )

    def resolve_args(self, args):
        return resolve_experiment_mode(args, ExperimentOptions)


class TestExperimentConfigOverrideParsing(
    ExperimentConfigOverrideTestCase,
    unittest.TestCase,
):
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
                "--weight-option",
                "LowRankDynamicWeightConfig",
            ]
        )

        _, _, _, overrides, search_overrides = self.resolve_args(args)

        self.assertEqual(overrides["num_epochs"], 30)
        self.assertEqual(overrides["callback_early_stopping_patience"], 0)
        self.assertEqual(overrides["learning_rate"], 1e-4)
        self.assertEqual(overrides["hidden_dim"], 128)
        self.assertIs(overrides["stack_activation"], ActivationOptions.GELU)
        self.assertIs(overrides["weight_option"], LowRankDynamicWeightConfig)
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

        _, _, _, overrides, search_overrides = self.resolve_args(args)

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

        _, _, _, plain_overrides, plain_search_overrides = self.resolve_args(
            plain_args
        )
        _, _, _, grouped_overrides, grouped_search_overrides = self.resolve_args(
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
                "weight_option=None,LowRankDynamicWeightConfig",
            ]
        )

        _, search_mode, _, _, search_overrides = self.resolve_args(args)

        self.assertIsInstance(search_mode, GridSearch)
        self.assertEqual(search_overrides["hidden_dim"], [64, 128])
        self.assertEqual(
            search_overrides["stack_activation"],
            [ActivationOptions.RELU, ActivationOptions.GELU],
        )
        self.assertEqual(
            search_overrides["weight_option"],
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

    def test_trainer_overrides_disable_early_stopping(self):
        experiment = Experiment(ExperimentOptions.SINGLE_MODEL_WEIGHT)

        trainer_config = experiment._load_trainer_config(
            {
                "callback_early_stopping_patience": 0,
                "trainer_devices": 1,
            }
        )
        callback_types = {type(callback) for callback in trainer_config["callbacks"]}

        self.assertNotIn(EarlyStopping, callback_types)
        self.assertIn(AdaptiveParameterMonitorCallback, callback_types)
        self.assertEqual(trainer_config["trainer_args"]["devices"], 1)


if __name__ == "__main__":
    unittest.main()
