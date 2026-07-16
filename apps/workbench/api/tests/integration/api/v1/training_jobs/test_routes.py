from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.settings import WorkbenchApiSettings
from tests.support import lifespan_client
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
)


class TrainingJobHttpTests(unittest.TestCase):
    """Public HTTP behavior backed by the application lifespan."""

    def test_training_api_response_does_not_expose_manager_internals(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
            )

            async def call_api() -> httpx.Response:
                async with lifespan_client(
                    test_app,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline", "gating"],
                            "datasets": ["Mnist"],
                            "overrides": {"hidden_dim": "128"},
                            "logFolder": "test_model",
                            "monitors": ["linear"],
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["pid"], 1234)
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, payload)

    def test_training_api_cancel_job_terminates_process(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(process),
            )
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
            )

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                async with lifespan_client(
                    test_app,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    create_response = await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "cancel_api",
                            "monitors": [],
                        },
                    )
                    job_id = create_response.json()["id"]
                    cancel_response = await client.post(
                        f"/training/jobs/{job_id}/cancel"
                    )
                    repeated_cancel_response = await client.post(
                        f"/training/jobs/{job_id}/cancel"
                    )
                    unknown_response = await client.post(
                        "/training/jobs/missing/cancel",
                        headers={"Idempotency-Key": uuid.uuid4().hex},
                    )
                    return (
                        create_response,
                        cancel_response,
                        repeated_cancel_response,
                        unknown_response,
                    )

            (
                create_response,
                cancel_response,
                repeated_cancel_response,
                unknown_response,
            ) = asyncio.run(call_api())

        self.assertEqual(create_response.status_code, 200, create_response.text)
        job_id = create_response.json()["id"]
        self.assertEqual(cancel_response.status_code, 200, cancel_response.text)
        payload = cancel_response.json()
        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(payload["events"][-1]["type"], "cancelled")
        self.assertEqual(payload["events"][-1]["status"], "cancelled")
        self.assertEqual(payload["events"][-1]["jobId"], job_id)
        self.assertEqual(payload["eventCounts"]["cancelled"], 1)
        self.assertEqual(repeated_cancel_response.status_code, 200)
        self.assertEqual(
            repeated_cancel_response.json()["eventCounts"]["cancelled"],
            1,
        )
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Skipped"],
        )
        self.assertEqual(payload["runPlan"]["summary"]["pendingRuns"], 0)
        self.assertEqual(payload["runPlan"]["summary"]["cancelledRuns"], 0)
        self.assertEqual(payload["runPlan"]["summary"]["skippedRuns"], 1)
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, payload)

        self.assertEqual(unknown_response.status_code, 400)
        self.assertEqual(
            unknown_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )

    def test_training_api_created_job_uses_safe_worker_command_paths(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            logs_root = root / "logs"
            runner = FakeRunner()
            manager = TrainingJobServiceHarness(
                root=jobs_root,
                logs_root=logs_root,
                runner=runner,
            )
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
            )

            async def call_api() -> httpx.Response:
                async with lifespan_client(
                    test_app,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline"],
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "path_safety",
                            "monitors": [],
                        },
                    )

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            self.assertEqual(len(job_id), 32)
            int(job_id, 16)
            job_root = jobs_root / job_id
            expected_payload = job_root / "payload.json"
            expected_progress = job_root / "progress.jsonl"
            expected_log = job_root / "training.log"

            self.assertEqual(len(runner.commands), 1)
            command = runner.commands[0]
            self.assertIsInstance(command, list)
            self.assertTrue(all(isinstance(part, str) for part in command))
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-m",
                    "emperor_workbench.training_jobs.worker",
                    "--payload",
                    str(expected_payload),
                    "--progress",
                    str(expected_progress),
                ],
            )
            self.assertEqual(runner.log_paths, [expected_log])

            resolved_job_root = job_root.resolve()
            for path in (expected_payload, expected_progress, expected_log):
                with self.subTest(path=path):
                    self.assertTrue(
                        path.resolve().is_relative_to(resolved_job_root),
                        f"{path} should stay under {resolved_job_root}",
                    )

    def test_training_api_get_unknown_job_returns_http_400(self) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
            )

            async def call_api() -> httpx.Response:
                async with lifespan_client(
                    test_app,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.get("/training/jobs/missing")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "Unknown training job 'missing'."},
        )


if __name__ == "__main__":
    unittest.main()
