from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.model_packages import model_package
from emperor.runs import (
    InvalidRunPlan,
    PlanningBudget,
    RunRequest,
    SubmittedRun,
    accept_run_plan,
)


def _linears_linear():
    package = model_package("linears/linear")
    if package is None:
        raise AssertionError("Expected the linears/linear Model Package.")
    return package


class RunsAcceptanceTests(unittest.TestCase):
    def test_submitted_rows_preserve_ids_and_order_without_rematerializing(
        self,
    ) -> None:
        plan = accept_run_plan(
            _linears_linear(),
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist", "Cifar10"),
            ),
            (
                SubmittedRun(
                    id="snapshot-2",
                    preset="gating",
                    dataset="Cifar10",
                    overrides={"hidden_dim": "128"},
                ),
                SubmittedRun(
                    id="preset-1",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={"stack_num_layers": "4"},
                ),
            ),
            budget=PlanningBudget(max_materialized_runs=2000),
        )

        self.assertEqual([run.id for run in plan.runs], ["snapshot-2", "preset-1"])
        self.assertEqual(
            [(run.preset, run.dataset) for run in plan.runs],
            [("gating", "Cifar10"), ("baseline", "Mnist")],
        )
        self.assertEqual(dict(plan.runs[0].overrides), {"HIDDEN_DIM": 128})
        self.assertEqual(
            dict(plan.runs[1].overrides),
            {"STACK_NUM_LAYERS": 4},
        )

    def test_submitted_plan_rejects_empty_duplicate_and_foreign_rows(self) -> None:
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
        )
        cases = (
            (
                (),
                "requires at least one training run",
            ),
            (
                (
                    SubmittedRun("same", "baseline", "Mnist"),
                    SubmittedRun("same", "baseline", "Mnist"),
                ),
                "duplicate run id 'same'",
            ),
            (
                (SubmittedRun("run-1", "gating", "Mnist"),),
                "unknown preset 'gating'",
            ),
            (
                (SubmittedRun("run-1", "baseline", "Cifar10"),),
                "unknown dataset 'Cifar10'",
            ),
        )
        for submitted, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(InvalidRunPlan, message):
                    accept_run_plan(
                        _linears_linear(),
                        request,
                        submitted,
                    )

    def test_submitted_plan_rejects_locked_override_even_when_equal(self) -> None:
        with self.assertRaisesRegex(
            InvalidRunPlan,
            "does not allow overriding locked fields: stack_gate_flag",
        ):
            accept_run_plan(
                _linears_linear(),
                RunRequest(
                    presets=("gating",),
                    datasets=("Mnist",),
                ),
                (
                    SubmittedRun(
                        "run-1",
                        "gating",
                        "Mnist",
                        {"stack_gate_flag": "true"},
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
