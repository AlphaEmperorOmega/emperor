from __future__ import annotations

import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.augmentations.adaptive_parameters.core.weight import (
    LowRankDynamicWeightConfig,
)
from emperor.experiments.base import GridSearch, RandomSearch
from emperor.model_packages import model_package
from emperor.runs import RunRequest, plan_runs
from models.linears.linear_adaptive.presets import ExperimentPreset
from models.package_cli import _search_spec, run_model_package_cli


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


def _linears_linear_adaptive():
    package = model_package("linears/linear_adaptive")
    if package is None:
        raise AssertionError("Expected the linears/linear_adaptive Model Package.")
    return package


class PackageCliRunsTests(unittest.TestCase):
    def test_invalid_log_folder_rejects_before_plan_materialization(self) -> None:
        args = SimpleNamespace(datasets=["mnist"], logdir="../invalid")
        mode = SimpleNamespace(
            experiment_task=None,
            preset=ExperimentPreset.BASELINE,
            selected_presets=None,
            search_mode=GridSearch(),
            search_keys=None,
            config_overrides={},
            search_overrides={},
            monitor_names=[],
        )
        parser = SimpleNamespace(parse_args=lambda: args)
        with (
            patch("models.package_cli.get_experiment_parser", return_value=parser),
            patch("models.package_cli.resolve_experiment_mode", return_value=mode),
            patch("models.package_cli.plan_runs") as planner,
            self.assertRaisesRegex(ValueError, "single relative folder"),
        ):
            run_model_package_cli(
                experiment_type=object,
                preset_type=ExperimentPreset,
                module_path="models.linears.linear_adaptive",
            )

        planner.assert_not_called()

    def test_search_adapter_preserves_axis_order_and_replaces_selected_values(
        self,
    ) -> None:
        search = _search_spec(
            SimpleNamespace(
                search_mode=GridSearch(),
                search_keys=["stack_activation", "hidden_dim"],
                search_overrides={
                    "hidden_dim": [64, 128],
                    "learning_rate": [0.001],
                },
            )
        )

        self.assertIsNotNone(search)
        assert search is not None
        self.assertEqual(
            [axis.key for axis in search.axes or ()],
            ["stack_activation", "hidden_dim", "learning_rate"],
        )
        self.assertIsNone(search.axes[0].values)
        self.assertEqual(search.axes[1].values, (64, 128))
        self.assertTrue(search.axes[1].allow_custom_values)

    def test_random_adapter_and_accepted_plan_do_not_serialize_caller_policy(
        self,
    ) -> None:
        request_search = _search_spec(
            SimpleNamespace(
                search_mode=RandomSearch(num_samples=2),
                search_keys=None,
                search_overrides={"hidden_dim": [64, 128]},
            )
        )

        self.assertIsNotNone(request_search)
        assert request_search is not None
        self.assertEqual(request_search.mode, "random")
        self.assertEqual(request_search.random_samples, 2)
        self.assertTrue(request_search.axes[0].allow_custom_values)

        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=request_search,
            ),
            random_source=random.Random(7),
        )

        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        self.assertFalse(plan.search.axes[0].allow_custom_values)

    def test_search_set_can_add_supported_runtime_default_axes(self) -> None:
        request_search = _search_spec(
            SimpleNamespace(
                search_mode=GridSearch(),
                search_keys=None,
                search_overrides={
                    "batch_size": [16, 32],
                    "num_epochs": [1, 2],
                },
            )
        )

        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=request_search,
            ),
        )

        self.assertEqual(
            [
                (run.overrides["BATCH_SIZE"], run.overrides["NUM_EPOCHS"])
                for run in plan.runs
            ],
            [(16, 1), (16, 2), (32, 1), (32, 2)],
        )
        self.assertEqual(
            [axis.key for axis in plan.search.axes],
            ["BATCH_SIZE", "NUM_EPOCHS"],
        )

    def test_search_set_keeps_custom_values_duplicates_and_class_values(
        self,
    ) -> None:
        custom_plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=_search_spec(
                    SimpleNamespace(
                        search_mode=GridSearch(),
                        search_keys=None,
                        search_overrides={"hidden_dim": [17, 17, 19]},
                    )
                ),
            ),
        )
        class_plan = plan_runs(
            _linears_linear_adaptive(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=_search_spec(
                    SimpleNamespace(
                        search_mode=GridSearch(),
                        search_keys=None,
                        search_overrides={
                            "weight_option": [None, LowRankDynamicWeightConfig]
                        },
                    )
                ),
            ),
        )

        self.assertEqual(
            [run.overrides["HIDDEN_DIM"] for run in custom_plan.runs],
            [17, 17, 19],
        )
        self.assertEqual(
            [run.overrides["WEIGHT_OPTION"] for run in class_plan.runs],
            [None, "LowRankDynamicWeightConfig"],
        )

    def test_package_cli_serializes_typed_fixed_class_override(self) -> None:
        args = SimpleNamespace(datasets=["mnist"], logdir=None)
        mode = SimpleNamespace(
            experiment_task=None,
            preset=ExperimentPreset.BASELINE,
            selected_presets=None,
            search_mode=None,
            search_keys=None,
            config_overrides={"weight_option": LowRankDynamicWeightConfig},
            search_overrides={},
            monitor_names=[],
        )
        parser = SimpleNamespace(parse_args=lambda: args)
        with (
            patch("models.package_cli.get_experiment_parser", return_value=parser),
            patch("models.package_cli.resolve_experiment_mode", return_value=mode),
            patch("models.package_cli.execute_runs", return_value=()) as execute,
        ):
            run_model_package_cli(
                experiment_type=object,
                preset_type=ExperimentPreset,
                module_path="models.linears.linear_adaptive",
            )

        execute.assert_called_once()
        _package, plan = execute.call_args.args
        self.assertEqual(
            dict(plan.overrides),
            {"WEIGHT_OPTION": "LowRankDynamicWeightConfig"},
        )
        self.assertEqual(
            dict(plan.runs[0].overrides),
            {"WEIGHT_OPTION": "LowRankDynamicWeightConfig"},
        )


if __name__ == "__main__":
    unittest.main()
