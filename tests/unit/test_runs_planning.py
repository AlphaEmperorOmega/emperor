from __future__ import annotations

import os
import random
import unittest
from dataclasses import FrozenInstanceError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.model_packages import config_key_to_model_param, model_package
from emperor.runs import (
    InvalidRunRequest,
    PlanningBudget,
    PlanTooLarge,
    RunRequest,
    SearchAxisSelection,
    SearchSpec,
    plan_runs,
)


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


class _ForbiddenRandom:
    def sample(self, population, k):
        raise AssertionError("Random selection must not run over budget.")

    def randrange(self, stop):
        raise AssertionError("Random selection must not run over budget.")


class RunsPlanningTests(unittest.TestCase):
    def test_cli_only_runtime_default_is_retained_by_executable_run(self) -> None:
        plan = plan_runs(
            _gpt_expert_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("WikiText2",),
                overrides={"num_experts": 7},
            ),
        )

        self.assertEqual(dict(plan.overrides), {"EXPERT_NUM_EXPERTS": 7})
        self.assertEqual(
            dict(plan.runs[0].overrides),
            {"EXPERT_NUM_EXPERTS": 7},
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
            {
                config_key_to_model_param(axis.key)
                for axis in plan.search.axes or ()
            },
        )
        equal_plan = plan_runs(
            package,
            RunRequest(
                presets=("top1-switch-aux",),
                datasets=("Multi30kDeEn",),
                overrides={"top_k": 1},
            ),
        )
        self.assertEqual(equal_plan.runs[0].overrides["EXPERT_TOP_K"], 1)
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
            "EXPERT_TOP_K.*locked by preset",
        ):
            plan_runs(
                package,
                RunRequest(
                    presets=("top1-switch-aux",),
                    datasets=("Multi30kDeEn",),
                    search=SearchSpec(
                        mode="grid",
                        axes=(
                            SearchAxisSelection("expert_top_k", (2,)),
                        ),
                    ),
                ),
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
                    overrides={"stack_layer_norm_position": "BEFORE"},
                    search=SearchSpec(mode="grid"),
                ),
                budget=PlanningBudget(max_materialized_runs=1),
            )

    def test_implicit_full_search_deduplicates_model_parameter_aliases(self) -> None:
        plan = plan_runs(
            _linears_linear(),
            RunRequest(
                presets=("baseline",),
                datasets=("Mnist",),
                search=SearchSpec(mode="random", random_samples=1),
            ),
            random_source=random.Random(7),
        )

        self.assertEqual(len(plan.runs), 1)
        self.assertIsNotNone(plan.search)
        assert plan.search is not None
        model_params = [
            config_key_to_model_param(axis.key)
            for axis in plan.search.axes or ()
        ]
        self.assertEqual(len(model_params), len(set(model_params)))
        self.assertIn("STACK_LAYER_NORM_POSITION", plan.runs[0].overrides)
        self.assertNotIn("LAYER_NORM_POSITION", plan.runs[0].overrides)

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

        self.assertEqual(equal_plan.runs[0].overrides["GATE_FLAG"], "true")
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
