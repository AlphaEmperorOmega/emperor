import unittest

from emperor.augmentations.adaptive_parameters.core.weight import (
    LowRankDynamicWeightConfig,
)
from emperor.base.options import ActivationOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import GridSearch
from models.linear_adaptive import ExperimentOptions
from models.linear_adaptive.presets import Experiment, ExperimentPresets
from models.parser import get_experiment_parser, resolve_experiment_mode


class TestExperimentConfigOverrides(unittest.TestCase):
    def parser(self):
        return get_experiment_parser(
            ExperimentOptions.names(),
            "models.linear_adaptive",
        )

    def resolve(self, args):
        return resolve_experiment_mode(args, ExperimentOptions)

    def test_named_config_flags_parse_supported_value_types(self):
        args = self.parser().parse_args(
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

        _, _, _, overrides, search_overrides = self.resolve(args)

        self.assertEqual(overrides["num_epochs"], 30)
        self.assertEqual(overrides["callback_early_stopping_patience"], 0)
        self.assertEqual(overrides["learning_rate"], 1e-4)
        self.assertEqual(overrides["hidden_dim"], 128)
        self.assertIs(overrides["stack_activation"], ActivationOptions.GELU)
        self.assertIs(overrides["weight_option"], LowRankDynamicWeightConfig)
        self.assertEqual(search_overrides, {})

    def test_config_marker_allows_grouped_config_overrides(self):
        args = self.parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--config",
                "--num-epochs",
                "30",
                "--callback-early-stopping-patience",
                "0",
            ]
        )

        _, _, _, overrides, search_overrides = self.resolve(args)

        self.assertEqual(overrides["num_epochs"], 30)
        self.assertEqual(overrides["callback_early_stopping_patience"], 0)
        self.assertEqual(search_overrides, {})

    def test_search_set_parses_lists_using_config_value_types(self):
        args = self.parser().parse_args(
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

        _, search_mode, _, _, search_overrides = self.resolve(args)

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
        args = self.parser().parse_args(
            [
                "--preset",
                "single-model-weight",
                "--search-set",
                "hidden_dim=64,128",
            ]
        )

        with self.assertRaises(ValueError):
            self.resolve(args)

    def test_fixed_override_and_search_axis_conflict_raises(self):
        args = self.parser().parse_args(
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
            self.resolve(args)

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

        self.assertEqual(trainer_config["callbacks"], [])
        self.assertEqual(trainer_config["trainer_args"]["devices"], 1)


if __name__ == "__main__":
    unittest.main()
