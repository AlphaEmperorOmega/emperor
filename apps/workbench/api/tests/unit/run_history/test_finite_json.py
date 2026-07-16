from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from emperor_workbench.run_history._artifacts import observe_run_artifacts


class RunHistoryFiniteJsonTests(unittest.TestCase):
    def test_existing_run_result_maps_non_finite_leaves_to_null_and_diagnoses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root / "experiment" / "linears" / "linear" / "run-1"
            run_dir.mkdir(parents=True)
            run_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "loss": math.nan,
                            "nested": [math.inf, {"score": -math.inf}],
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertLogs(
                "emperor_workbench.run_history._artifacts",
                level="WARNING",
            ) as captured:
                metrics = observe_run_artifacts(run_dir, logs_root).metrics()

        self.assertEqual(
            metrics,
            {"loss": None, "nested": [None, {"score": None}]},
        )
        diagnostic = "\n".join(captured.output)
        self.assertIn("experiment/linears/linear/run-1", diagnostic)
        self.assertIn("$.metrics.loss", diagnostic)
        self.assertIn("$.metrics.nested[0]", diagnostic)
        self.assertIn("$.metrics.nested[1].score", diagnostic)


if __name__ == "__main__":
    unittest.main()
