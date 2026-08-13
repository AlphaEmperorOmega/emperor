from __future__ import annotations

import os
import random
import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import ModuleType
from typing import Any
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from model_runtime.inspection import InspectionError, configuration_schema
from model_runtime.packages import (
    ModelMetadata,
    ModelPackage,
    config_key_to_model_param,
)
from model_runtime.runs import (
    InvalidRunRequest,
    PlanningBudget,
    PlanTooLarge,
    RunRequest,
    SearchAxisSelection,
    SearchSpec,
    plan_runs,
)
from models.catalog import model_package
from models.linears.linear import config as linears_linear_config
from models.linears.linear import dataset_options as linears_linear_datasets
from models.linears.linear import monitor_options as linears_linear_monitors


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


def _transformer_expert_linear():
    package = model_package("transformer/expert_linear")
    if package is None:
        raise AssertionError("Expected the transformer/expert_linear Model Package.")
    return package


def _gpt_expert_linear():
    package = model_package("gpt/expert_linear")
    if package is None:
        raise AssertionError("Expected the gpt/expert_linear Model Package.")
    return package


def _gpt_linear_adaptive():
    package = model_package("gpt/linear_adaptive")
    if package is None:
        raise AssertionError("Expected the gpt/linear_adaptive Model Package.")
    return package


class _SearchMetadataAdapter:
    def __init__(self, package: ModelPackage, metadata: ModelMetadata) -> None:
        self._package = package
        self._metadata = metadata

    def load_metadata(self) -> ModelMetadata:
        return self._metadata

    def load_runtime_options_type(self) -> type[Any]:
        return self._package.runtime_options_type

    def bind_runtime_defaults(self, values: Mapping[str, object] | None) -> Any:
        return self._package.bind_runtime_defaults(values)

    def load_preset_type(self) -> type[Any]:
        return self._package.preset_type

    def load_presets(self) -> Any:
        return self._package.presets

    def build_configuration(
        self,
        presets: Any,
        preset: Any,
        dataset: type[Any],
        **kwargs: Any,
    ) -> Any:
        del presets
        return self._package.build_configuration(preset, dataset, **kwargs)

    def build_model(self, configuration: Any) -> Any:
        return self._package.build_model(configuration)

    def build_experiment(
        self,
        preset: Any,
        *,
        experiment_task: Any,
        model_package: ModelPackage,
        run_artifacts: Any,
    ) -> Any:
        del model_package
        return self._package.build_experiment(
            preset,
            experiment_task=experiment_task,
            run_artifacts=run_artifacts,
        )


def _linears_linear_with_duplicate_implicit_axes() -> ModelPackage:
    package = _linears_linear()
    search_space = ModuleType("models.linears.linear.search_space")
    search_space.SEARCH_SPACE_HIDDEN_DIM = [64]
    setattr(search_space, "SEARCH_SPACE_HIDDEN-DIM", [128])
    metadata = ModelMetadata(
        identity=package.identity,
        runtime_defaults=linears_linear_config,
        dataset_options=linears_linear_datasets,
        monitor_options_source=linears_linear_monitors,
        search_space=search_space,
    )
    return ModelPackage(
        package.identity,
        _SearchMetadataAdapter(package, metadata),
    )


class _ForbiddenRandom:
    def sample(self, population, k):
        raise AssertionError("Random selection must not run over budget.")

    def randrange(self, stop):
        raise AssertionError("Random selection must not run over budget.")


class RunsPlanningTests(unittest.TestCase):
    def test_runs_planning_does_not_require_inspection_presentation_metadata(
        self,
    ) -> None:
        package = _linears_linear()
        catalog_metadata = package._adapter.load_metadata()
        legacy_metadata = ModelMetadata(
            package.identity,
            catalog_metadata._runtime_defaults_source,
            catalog_metadata._dataset_options_source,
            catalog_metadata._monitor_options_source,
            catalog_metadata._search_space_source,
        )
        package_without_headings = ModelPackage(
            package.identity,
            _SearchMetadataAdapter(package, legacy_metadata),
            package.inspection_construction_limits,
        )
        with patch(
            "model_runtime.packages.metadata.configuration_field_metadata",
            return_value={},
        ):
            plan = plan_runs(
                package_without_headings,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    overrides={"hidden_dim": 64},
                ),
            )

        self.assertEqual(dict(plan.runs[0].overrides), {"HIDDEN_DIM": 64})
        with self.assertRaisesRegex(
            InspectionError,
            "missing source heading metadata",
        ):
            configuration_schema(package_without_headings)

    def test_planning_budget_requires_positive_plain_integers_or_none(self) -> None:
        for field_name in (
            "max_axes",
            "max_values_per_axis",
            "max_materialized_runs",
        ):
            for invalid_value in (True, False, 0, -1, 1.5, "2"):
                with self.subTest(field=field_name, value=invalid_value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be a positive integer or None",
                    ):
                        PlanningBudget(**{field_name: invalid_value})

    def test_cli_only_runtime_default_is_retained_by_executable_run(self) -> None:
        plan = plan_runs(
            _gpt_expert_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("WikiText2",),
                overrides={"num_experts": 7},
            ),
        )

        self.assertEqual(dict(plan.overrides), {"NUM_EXPERTS": 7})
        self.assertEqual(
            dict(plan.runs[0].overrides),
            {"NUM_EXPERTS": 7},
        )

    def test_visible_fixed_parameter_order_preserves_schema_order(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                overrides={"hidden_dim": 64, "input_dim": 32},
            ),
        )

        self.assertEqual(
            [parameter.key for parameter in plan.runs[0].parameters],
            ["INPUT_DIM", "HIDDEN_DIM"],
        )

    def test_expert_lock_alias_is_excluded_and_rejects_explicit_search(self) -> None:
        package = _transformer_expert_linear()
        plan = plan_runs(
            package,
            RunRequest(
                presets=("top1-switch-aux",),
                datasets=("Multi30kDeEn",),
                search=SearchSpec(mode="random", random_samples=1),
            ),
            random_source=random.Random(11),
        )

        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        self.assertNotIn(
            "top_k",
            {config_key_to_model_param(axis.key) for axis in plan.search.axes or ()},
        )
        equal_plan = plan_runs(
            package,
            RunRequest(
                presets=("top1-switch-aux",),
                datasets=("Multi30kDeEn",),
                overrides={"top_k": 1},
            ),
        )
        self.assertEqual(equal_plan.runs[0].overrides["TOP_K"], 1)
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "does not allow overriding locked fields: top_k",
        ):
            plan_runs(
                package,
                RunRequest(
                    presets=("top1-switch-aux",),
                    datasets=("Multi30kDeEn",),
                    overrides={"top_k": 2},
                ),
            )
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "TOP_K.*locked by preset",
        ):
            plan_runs(
                package,
                RunRequest(
                    presets=("top1-switch-aux",),
                    datasets=("Multi30kDeEn",),
                    search=SearchSpec(
                        mode="grid",
                        axes=(SearchAxisSelection("top_k", (2,)),),
                    ),
                ),
            )

    def test_implicit_search_preserves_present_legacy_none_lock(self) -> None:
        package = _linears_linear()
        with patch.object(
            ModelPackage,
            "preset_locks",
            return_value={"HIDDEN_DIM": None},
        ):
            plan = plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    search=SearchSpec(mode="random", random_samples=1),
                ),
                random_source=random.Random(7),
                budget=PlanningBudget.unlimited(),
            )

        assert plan.search is not None
        self.assertNotIn(
            "HIDDEN_DIM",
            {axis.key for axis in plan.search.axes or ()},
        )

    def test_implicit_search_honors_axis_and_value_budgets(self) -> None:
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
            search=SearchSpec(mode="grid"),
        )

        with self.assertRaisesRegex(PlanTooLarge, "at most 1 selected axes"):
            plan_runs(
                _linears_linear(),
                request,
                budget=PlanningBudget(max_axes=1),
            )
        with self.assertRaisesRegex(
            PlanTooLarge,
            "LEARNING_RATE.*at most 2 selected values",
        ):
            plan_runs(
                _linears_linear(),
                request,
                budget=PlanningBudget(max_values_per_axis=2),
            )

    def test_search_validation_precedence_is_stable(self) -> None:
        package = _linears_linear()
        request_values = {
            "presets": ("baseline",),
            "datasets": ("Mnist",),
        }

        with self.assertRaisesRegex(InvalidRunRequest, "mode must be 'grid'"):
            plan_runs(
                package,
                RunRequest(
                    **request_values,
                    search=SearchSpec(mode="invalid", axes=()),
                ),
                budget=PlanningBudget(max_axes=1),
            )

        with self.assertRaisesRegex(InvalidRunRequest, "at least one selected axis"):
            plan_runs(
                package,
                RunRequest(
                    **request_values,
                    search=SearchSpec(mode="grid", axes=()),
                ),
                budget=PlanningBudget(max_axes=1),
            )

        with self.assertRaisesRegex(PlanTooLarge, "at most 1 selected axes"):
            plan_runs(
                package,
                RunRequest(
                    **request_values,
                    search=SearchSpec(
                        mode="grid",
                        axes=(
                            SearchAxisSelection("unknown-a", (1,)),
                            SearchAxisSelection("unknown-b", (1,)),
                        ),
                    ),
                ),
                budget=PlanningBudget(max_axes=1),
            )

        with self.assertRaisesRegex(PlanTooLarge, "at most 1 selected values"):
            plan_runs(
                package,
                RunRequest(
                    **request_values,
                    search=SearchSpec(
                        mode="grid",
                        axes=(SearchAxisSelection("unknown", (1, 2)),),
                    ),
                ),
                budget=PlanningBudget(max_values_per_axis=1),
            )

        locked_package = _transformer_expert_linear()
        locked_request_values = {
            "presets": ("top1-switch-aux",),
            "datasets": ("Multi30kDeEn",),
        }
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "Invalid search value for axis 'TOP_K'",
        ):
            plan_runs(
                locked_package,
                RunRequest(
                    **locked_request_values,
                    search=SearchSpec(
                        mode="grid",
                        axes=(SearchAxisSelection("top_k", ("invalid",)),),
                    ),
                ),
            )

        with self.assertRaisesRegex(InvalidRunRequest, "TOP_K.*locked by preset"):
            plan_runs(
                locked_package,
                RunRequest(
                    **locked_request_values,
                    search=SearchSpec(
                        mode="grid",
                        axes=(SearchAxisSelection("top_k", (999,)),),
                    ),
                ),
            )

    def test_default_budget_rejects_implicit_search_before_random_selection(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            PlanTooLarge,
            "at most 16 selected axes",
        ):
            plan_runs(
                _gpt_linear_adaptive(),
                RunRequest(
                    presets=("baseline",),
                    datasets=("WikiText2",),
                    search=SearchSpec(mode="random", random_samples=1),
                ),
                random_source=_ForbiddenRandom(),
            )

    def test_default_budget_rejects_oversized_grid_before_materialization(
        self,
    ) -> None:
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
            search=SearchSpec(
                mode="grid",
                axes=(
                    SearchAxisSelection(
                        "hidden_dim",
                        tuple(range(45)),
                        allow_custom_values=True,
                    ),
                    SearchAxisSelection(
                        "stack_num_layers",
                        tuple(range(1, 46)),
                        allow_custom_values=True,
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            PlanTooLarge,
            "2025 planned runs exceeds 2000",
        ):
            plan_runs(_linears_linear(), request)

    def test_unlimited_budget_requires_explicit_opt_in(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(
                    mode="grid",
                    axes=(
                        SearchAxisSelection(
                            "hidden_dim",
                            tuple(range(45)),
                            allow_custom_values=True,
                        ),
                        SearchAxisSelection(
                            "stack_num_layers",
                            tuple(range(1, 46)),
                            allow_custom_values=True,
                        ),
                    ),
                ),
            ),
            budget=PlanningBudget.unlimited(),
        )

        self.assertEqual(len(plan.runs), 2_025)

    def test_invalid_falsey_planning_budgets_are_rejected(self) -> None:
        request = RunRequest(presets=("baseline",), datasets=("Mnist",))

        for invalid in (False, 0, "unlimited"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(TypeError, "budget must be a PlanningBudget"),
            ):
                plan_runs(_linears_linear(), request, budget=invalid)

    def test_implicit_search_does_not_strip_fixed_override_from_locked_preset(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "post-norm.*locked fields.*layer_norm_position",
        ):
            plan_runs(
                _linears_linear_adaptive(),
                RunRequest(
                    presets=("baseline", "post-norm"),
                    datasets=("Mnist",),
                    overrides={"layer_norm_position": "BEFORE"},
                    search=SearchSpec(mode="grid"),
                ),
                budget=PlanningBudget(
                    max_axes=None,
                    max_values_per_axis=None,
                    max_materialized_runs=1,
                ),
            )

    def test_run_plan_preserves_each_presets_effective_search_provenance(
        self,
    ) -> None:
        plan = plan_runs(
            _linears_linear_adaptive(),
            RunRequest(
                presets=("baseline", "post-norm"),
                datasets=("Mnist",),
                overrides={"layer_norm_position": "AFTER"},
                search=SearchSpec(mode="random", random_samples=1),
            ),
            random_source=random.Random(17),
            budget=PlanningBudget(
                max_axes=None,
                max_values_per_axis=None,
                max_materialized_runs=2,
            ),
        )

        self.assertEqual(
            [entry.preset for entry in plan.preset_searches],
            ["baseline", "post-norm"],
        )
        baseline_search = plan.search_for_preset("baseline")
        post_norm_search = plan.search_for_preset("post-norm")
        self.assertIs(plan.search, baseline_search)
        self.assertIsNotNone(baseline_search)
        self.assertIsNotNone(post_norm_search)
        assert baseline_search is not None
        assert post_norm_search is not None
        self.assertEqual(len(baseline_search.axes or ()), 31)
        self.assertEqual(len(post_norm_search.axes or ()), 30)
        self.assertIn(
            "LAYER_NORM_POSITION",
            {axis.key for axis in baseline_search.axes or ()},
        )
        self.assertNotIn(
            "LAYER_NORM_POSITION",
            {axis.key for axis in post_norm_search.axes or ()},
        )

        post_norm_run = next(run for run in plan.runs if run.preset == "post-norm")
        layer_norm_parameter = next(
            parameter
            for parameter in post_norm_run.parameters
            if parameter.key == "LAYER_NORM_POSITION"
        )
        self.assertEqual(layer_norm_parameter.value, "AFTER")
        self.assertEqual(layer_norm_parameter.source, "override")
        with self.assertRaisesRegex(KeyError, "unknown"):
            plan.search_for_preset("unknown")

    def test_implicit_full_search_deduplicates_model_parameter_aliases(self) -> None:
        plan = plan_runs(
            _linears_linear_with_duplicate_implicit_axes(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(mode="grid"),
            ),
        )

        self.assertEqual(len(plan.runs), 1)
        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        self.assertEqual(
            [(axis.key, axis.values) for axis in plan.search.axes or ()],
            [("HIDDEN-DIM", (128,))],
        )
        self.assertEqual(dict(plan.runs[0].overrides), {"HIDDEN-DIM": 128})

    def test_supported_experiment_tasks_resolve_only_package_datasets(self) -> None:
        cases = (
            ("linears/linear", "image-classification", "Mnist"),
            ("gpt/linear", "causal-language-modeling", "WikiText2"),
            ("transformer/linear", "text-translation", "Multi30kDeEn"),
            (
                "bert/linear",
                "bert-pretraining",
                "PennTreebankBertPretraining",
            ),
        )
        for package_key, task, dataset in cases:
            with self.subTest(package=package_key, task=task):
                package = model_package(package_key)
                if package is None:
                    self.fail(f"Expected the {package_key} Model Package.")
                plan = plan_runs(
                    package,
                    RunRequest(
                        presets=("baseline",),
                        datasets=(dataset,),
                        experiment_task=task,
                    ),
                )
                self.assertEqual(plan.experiment_task, task)
                self.assertEqual(plan.datasets, (dataset,))
                self.assertEqual(plan.runs[0].dataset, dataset)

        package = model_package("gpt/linear")
        if package is None:
            self.fail("Expected the gpt/linear Model Package.")
        with self.assertRaisesRegex(InvalidRunRequest, "Unknown dataset 'Mnist'"):
            plan_runs(
                package,
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    experiment_task="causal-language-modeling",
                ),
            )

    def test_no_search_tracer_materializes_one_immutable_run(self) -> None:
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
            overrides={"hidden_dim": "128"},
        )

        plan = plan_runs(
            _linears_linear(),
            request,
            budget=PlanningBudget(max_materialized_runs=1),
        )

        self.assertEqual(plan.identity.catalog_key, "linears/linear")
        self.assertEqual(plan.presets, ("baseline",))
        self.assertEqual(plan.experiment_task, "image-classification")
        self.assertEqual(plan.datasets, ("Mnist",))
        self.assertEqual(dict(plan.overrides), {"HIDDEN_DIM": 128})
        self.assertIsNone(plan.search)
        self.assertEqual(len(plan.runs), 1)
        run = plan.runs[0]
        self.assertEqual(run.id, "run-0001")
        self.assertEqual(run.preset, "baseline")
        self.assertEqual(run.experiment_task, "image-classification")
        self.assertEqual(run.dataset, "Mnist")
        self.assertEqual(dict(run.overrides), {"HIDDEN_DIM": "128"})
        self.assertEqual(run.parameters[0].source, "override")
        with self.assertRaises(FrozenInstanceError):
            run.dataset = "Cifar10"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            plan.overrides["HIDDEN_DIM"] = 64  # type: ignore[index]

    def test_multi_preset_and_dataset_order_is_stable(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline", "gating", "baseline"),
                datasets=("Mnist", "Cifar10", "Mnist"),
            ),
        )

        self.assertEqual(plan.presets, ("baseline", "gating"))
        self.assertEqual(plan.datasets, ("Mnist", "Cifar10"))
        self.assertEqual(
            [(run.preset, run.dataset) for run in plan.runs],
            [
                ("baseline", "Mnist"),
                ("baseline", "Cifar10"),
                ("gating", "Mnist"),
                ("gating", "Cifar10"),
            ],
        )
        self.assertEqual(
            [run.id for run in plan.runs],
            ["run-0001", "run-0002", "run-0003", "run-0004"],
        )

    def test_grid_search_preserves_axis_and_value_order(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(
                    mode="grid",
                    axes=(
                        SearchAxisSelection("hidden_dim", (64, 128)),
                        SearchAxisSelection(
                            "stack_activation",
                            ("RELU", "GELU"),
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(
            [
                (
                    run.overrides["HIDDEN_DIM"],
                    run.overrides["STACK_ACTIVATION"],
                )
                for run in plan.runs
            ],
            [
                (64, "RELU"),
                (64, "GELU"),
                (128, "RELU"),
                (128, "GELU"),
            ],
        )
        self.assertTrue(
            all(
                parameter.source == "search"
                for run in plan.runs
                for parameter in run.parameters
            )
        )

    def test_normalized_search_preserves_custom_value_authorization(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(
                    mode="grid",
                    axes=(
                        SearchAxisSelection(
                            "hidden_dim",
                            (65,),
                            allow_custom_values=True,
                        ),
                    ),
                ),
            ),
        )

        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        self.assertEqual(
            plan.search.axes,
            (
                SearchAxisSelection(
                    "HIDDEN_DIM",
                    (65,),
                    allow_custom_values=True,
                ),
            ),
        )
        effective_search = plan.search_for_preset("baseline")
        self.assertEqual(effective_search, plan.search)

        replayed = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=effective_search,
            ),
        )
        self.assertEqual(replayed.search_for_preset("baseline"), effective_search)

    def test_explicit_duplicate_axis_alias_keeps_position_and_last_values(
        self,
    ) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(
                    mode="grid",
                    axes=(
                        SearchAxisSelection("hidden_dim", (64,)),
                        SearchAxisSelection("stack_activation", ("RELU",)),
                        SearchAxisSelection("HIDDEN-DIM", (128,)),
                    ),
                ),
            ),
        )

        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        self.assertEqual(
            [(axis.key, axis.values) for axis in plan.search.axes or ()],
            [
                ("HIDDEN_DIM", (128,)),
                ("STACK_ACTIVATION", ("RELU",)),
            ],
        )
        self.assertEqual(
            dict(plan.runs[0].overrides),
            {"HIDDEN_DIM": 128, "STACK_ACTIVATION": "RELU"},
        )

    def test_explicit_duplicate_axis_validates_values_before_replacement(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "Invalid search value for axis 'HIDDEN_DIM'",
        ):
            plan_runs(
                _linears_linear(),
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    search=SearchSpec(
                        mode="grid",
                        axes=(
                            SearchAxisSelection(
                                "hidden_dim",
                                ("not-an-integer",),
                            ),
                            SearchAxisSelection("HIDDEN-DIM", (128,)),
                        ),
                    ),
                ),
            )

    def test_seeded_random_search_resamples_per_preset_dataset_block(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist", "Cifar10"),
                search=SearchSpec(
                    mode="random",
                    axes=(
                        SearchAxisSelection("hidden_dim", (64, 128)),
                        SearchAxisSelection(
                            "stack_activation",
                            ("RELU", "GELU"),
                        ),
                    ),
                    random_samples=3,
                ),
            ),
            random_source=random.Random(13),
        )

        self.assertEqual(
            [
                (
                    run.preset,
                    run.dataset,
                    run.overrides["HIDDEN_DIM"],
                    run.overrides["STACK_ACTIVATION"],
                )
                for run in plan.runs
            ],
            [
                ("baseline", "Mnist", 128, "RELU"),
                ("baseline", "Mnist", 64, "GELU"),
                ("baseline", "Mnist", 64, "RELU"),
                ("baseline", "Cifar10", 64, "GELU"),
                ("baseline", "Cifar10", 128, "RELU"),
                ("baseline", "Cifar10", 64, "RELU"),
                ("gating", "Mnist", 64, "GELU"),
                ("gating", "Mnist", 128, "RELU"),
                ("gating", "Mnist", 64, "RELU"),
                ("gating", "Cifar10", 64, "GELU"),
                ("gating", "Cifar10", 64, "RELU"),
                ("gating", "Cifar10", 128, "RELU"),
            ],
        )

    def test_plan_budget_rejects_before_random_selection(self) -> None:
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
            search=SearchSpec(
                mode="grid",
                axes=(
                    SearchAxisSelection(
                        "hidden_dim",
                        tuple(range(50)),
                        allow_custom_values=True,
                    ),
                    SearchAxisSelection(
                        "stack_num_layers",
                        tuple(range(1, 51)),
                        allow_custom_values=True,
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            PlanTooLarge,
            "2500 planned runs exceeds 2000",
        ):
            plan_runs(
                _linears_linear(),
                request,
                budget=PlanningBudget(
                    max_axes=16,
                    max_values_per_axis=50,
                    max_materialized_runs=2000,
                ),
            )

        random_request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist", "Cifar10"),
            search=SearchSpec(
                mode="random",
                axes=request.search.axes,
                random_samples=2000,
            ),
        )
        with self.assertRaisesRegex(
            PlanTooLarge,
            "4000 planned runs exceeds 2000",
        ):
            plan_runs(
                _linears_linear(),
                random_request,
                random_source=_ForbiddenRandom(),
                budget=PlanningBudget(
                    max_axes=16,
                    max_values_per_axis=50,
                    max_materialized_runs=2000,
                ),
            )

    def test_equal_locked_value_is_a_semantic_noop_but_conflict_rejects(
        self,
    ) -> None:
        equal_plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("gating",),
                datasets=("Mnist",),
                overrides={"stack_gate_flag": "true"},
            ),
        )

        self.assertEqual(equal_plan.runs[0].overrides["STACK_GATE_FLAG"], "true")
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "does not allow overriding locked fields: stack_gate_flag",
        ):
            plan_runs(
                _linears_linear(),
                RunRequest(
                    presets=("gating",),
                    datasets=("Mnist",),
                    overrides={"stack_gate_flag": "false"},
                ),
            )

    def test_random_search_requires_explicit_random_source(self) -> None:
        with self.assertRaisesRegex(
            InvalidRunRequest,
            "requires an explicit random source",
        ):
            plan_runs(
                _linears_linear(),
                RunRequest(
                    presets=("baseline",),
                    datasets=("Mnist",),
                    search=SearchSpec(
                        mode="random",
                        axes=(SearchAxisSelection("hidden_dim", (64, 128)),),
                        random_samples=1,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
