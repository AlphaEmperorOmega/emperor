from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path

import httpx

from model_runtime.runs.artifacts import write_run_result
from model_runtime.runs.progress import JsonlTrainingProgressCallback
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.main import create_app
from workbench.backend.run_history.artifacts import observe_run_artifacts


class FiniteJsonBoundaryTests(unittest.TestCase):
    def test_run_plan_rejects_non_finite_nested_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(WorkbenchApiSettings(logs_root=str(Path(tmp) / "logs")))

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.post(
                        "/training/run-plan",
                        content=json.dumps(
                            {
                                "modelType": "linears",
                                "model": "linear",
                                "preset": "baseline",
                                "datasets": ["Mnist"],
                                "overrides": {"dropout": math.nan},
                                "logFolder": "finite-json",
                            }
                        ),
                        headers={"content-type": "application/json"},
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 422, response.text)

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
                "workbench.backend.run_history.artifacts",
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

    def test_new_run_artifacts_and_progress_never_persist_non_finite_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "run" / "result.json"
            with self.assertRaises(ValueError):
                write_run_result(result_path.parent, {"metrics": {"loss": math.nan}})
            self.assertFalse(result_path.exists())

            progress_path = root / "job" / "progress.jsonl"
            progress = JsonlTrainingProgressCallback(progress_path)
            with self.assertRaises(ValueError):
                progress.write_event({"type": "step", "metrics": {"loss": math.inf}})
            self.assertFalse(progress_path.exists())


if __name__ == "__main__":
    unittest.main()
