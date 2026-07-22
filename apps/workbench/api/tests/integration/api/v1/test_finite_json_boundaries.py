from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
import uuid
from pathlib import Path

import httpx
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from model_runtime.runs.progress import JsonlRunProgress

from emperor_workbench.api import create_app
from emperor_workbench.settings import WorkbenchApiSettings


class FiniteJsonBoundaryTests(unittest.TestCase):
    def test_run_plan_rejects_non_finite_nested_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(root / "logs"),
                    snapshots_root=str(root / "snapshots"),
                    state_root=str(root / "state"),
                    training_cancellation_mode="process-group",
                )
            )

            async def call_api() -> httpx.Response:
                async with app.router.lifespan_context(app):
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

    def test_new_run_artifacts_and_progress_never_persist_non_finite_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "run" / "result.json"
            with self.assertRaises(ValueError):
                FilesystemRunArtifacts(root=root).write_result(
                    result_path.parent,
                    {"metrics": {"loss": math.nan}},
                )
            self.assertFalse(result_path.exists())

            progress_path = root / "job" / "progress.jsonl"
            progress = JsonlRunProgress(progress_path)
            with self.assertRaises(ValueError):
                progress.write_event({"type": "step", "metrics": {"loss": math.inf}})
            self.assertFalse(progress_path.exists())


if __name__ == "__main__":
    unittest.main()
