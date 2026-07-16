from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from emperor_workbench.run_plans import RunPlanWorkerAcceptance
from tests.support.model_packages import project_adapter_client
from tests.support.run_plans import worker_payload


class TrainingWorkerAcceptanceLimitTests(unittest.TestCase):
    def test_worker_rejects_limits_before_copying_private_plan_or_search(self) -> None:
        package = project_adapter_client().package("linears/linear")
        payload = copy.deepcopy(worker_payload())
        payload["runPlan"]["runs"] = [
            copy.deepcopy(payload["runPlan"]["runs"][0]) for _index in range(2001)
        ]
        with (
            patch(
                "emperor_workbench.run_plans._worker_acceptance.SubmittedRun"
            ) as submitted_run,
            self.assertRaisesRegex(ValueError, "submitted runs exceeds 2000"),
        ):
            RunPlanWorkerAcceptance.accept(package, payload)
            submitted_run.assert_not_called()

        for search, message in (
            (
                {
                    "mode": "grid",
                    "values": {f"axis_{index}": [1] for index in range(17)},
                },
                "at most 16 selected axes",
            ),
            (
                {
                    "mode": "grid",
                    "values": {"hidden_dim": list(range(51))},
                },
                "at most 50 selected values",
            ),
            (
                {
                    "mode": "random",
                    "values": {"hidden_dim": [64]},
                    "randomSamples": 2001,
                },
                "between 1 and 2000",
            ),
        ):
            with self.subTest(message=message):
                payload = copy.deepcopy(worker_payload())
                payload["runPlan"]["search"] = copy.deepcopy(search)
                with (
                    patch(
                        "emperor_workbench.run_plans._worker_acceptance.SubmittedRun"
                    ) as submitted_run,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    RunPlanWorkerAcceptance.accept(package, payload)
                submitted_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
