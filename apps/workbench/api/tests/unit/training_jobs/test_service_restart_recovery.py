from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobFailure
from tests.support import lifespan_client
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
    create_app_with_training_service,
)
from tests.unit.training_jobs._support import (
    TrainingJobServiceHarness,
    create_restart_limitation_job,
)


class TrainingJobRestartRecoveryTests(unittest.TestCase):
    _create_restart_limitation_job = staticmethod(create_restart_limitation_job)

    def test_training_job_restart_behavior_fresh_manager_gets_persisted_disk_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, created_payload, _ = self._create_restart_limitation_job(root)
            job_id = str(created_payload["id"])
            job_root = root / "jobs" / job_id
            self.assertTrue(job_root.joinpath("payload.json").is_file())
            self.assertTrue(job_root.joinpath("progress.jsonl").is_file())
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            payload = fresh_manager.get_job_payload(job_id)

        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline"])
        self.assertEqual(payload["datasets"], ["Mnist"])
        self.assertEqual(payload["logFolder"], "restart_limitation")
        self.assertEqual(payload["pid"], 1234)

    def test_restart_fresh_manager_cannot_cancel_disk_job_without_live_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            _, created_payload, _ = self._create_restart_limitation_job(
                root,
                process=process,
                log_folder="restart_cancel_limitation",
            )
            job_id = str(created_payload["id"])
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            with self.assertRaises(TrainingJobFailure) as context:
                fresh_manager.cancel_job_payload(job_id)

            self.assertFalse(process.terminated)
            self.assertIsNone(process.exit_code)

        self.assertEqual(
            str(context.exception),
            f"Training job '{job_id}' has no live process handle.",
        )

    def test_restart_fresh_manager_preserves_unknown_disk_job_blocker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_active_limitation",
            )
            job_id = str(created_payload["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "restart_active_limitation",
                    }
                ],
            )
            self.assertTrue((root / "jobs" / job_id / "payload.json").is_file())
            self.assertTrue((root / "jobs" / job_id / "progress.jsonl").is_file())
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            self.assertEqual(
                fresh_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "restart_active_limitation",
                    }
                ],
            )

    def test_training_job_restart_behavior_fresh_manager_reconstructs_disk_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_behavior",
            )
            job_id = str(created_payload["id"])
            job = original_manager.jobs[job_id]
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/restart_behavior/linear/baseline/Mnist/version_0"
            original_manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            )
            original_manager.runtime._write_event(
                job,
                {
                    "type": "validation",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 4,
                    "metrics": {"validation/accuracy": 0.75},
                    "logDir": log_dir,
                },
            )
            job_root = root / "jobs" / job_id
            self.assertTrue(job_root.joinpath("payload.json").is_file())
            self.assertTrue(job_root.joinpath("progress.jsonl").is_file())
            original_payload = original_manager.get_job_payload(job_id)

            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = fresh_manager.get_job_payload(job_id)

        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline"])
        self.assertEqual(payload["datasets"], ["Mnist"])
        self.assertEqual(payload["logFolder"], "restart_behavior")
        self.assertEqual(payload["plannedRunCount"], 1)
        self.assertEqual(
            [event["type"] for event in payload["events"]],
            ["job_started", "dataset_started", "validation"],
        )
        self.assertEqual(payload["currentPreset"], "baseline")
        self.assertEqual(payload["currentDataset"], "Mnist")
        self.assertEqual(payload["step"], 4)
        self.assertEqual(payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(payload["logDir"], log_dir)
        self.assertEqual(payload["runPlan"]["runs"][0]["status"], "Running")
        self.assertEqual(payload["runPlan"]["summary"]["runningRuns"], 1)
        for key in (
            "runPlan",
            "currentPreset",
            "currentDataset",
            "epoch",
            "step",
            "metrics",
            "logDir",
            "events",
            "eventCount",
            "eventCounts",
            "eventsTruncated",
            "clusterGrowth",
            "logTail",
            "resultLinks",
        ):
            with self.subTest(projection_field=key):
                self.assertEqual(payload[key], original_payload[key])

    def test_restart_fresh_manager_uses_completed_progress_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_completed_behavior",
            )
            job_id = str(created_payload["id"])
            job = original_manager.jobs[job_id]
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/restart_completed_behavior/linear/baseline/Mnist/version_0"
            original_manager.runtime._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.8},
                    "logDir": log_dir,
                },
            )
            original_manager.runtime._write_event(
                job,
                {
                    "type": "completed",
                    "status": "completed",
                    "jobId": job_id,
                    "preset": "baseline",
                    "presets": ["baseline"],
                },
            )

            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = fresh_manager.get_job_payload(job_id)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["exitCode"], 0)
            self.assertEqual(payload["eventCounts"]["completed"], 1)
            self.assertEqual(payload["runPlan"]["summary"]["completedRuns"], 1)
            self.assertEqual(fresh_manager.active_job_payloads(), [])
            self.assertEqual(fresh_manager.jobs[job_id].status, "completed")
            self.assertEqual(fresh_manager.jobs[job_id].exit_code, 0)

    def test_restart_fresh_manager_lists_unknown_disk_job_as_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_active_behavior",
            )
            job_id = str(created_payload["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "restart_active_behavior",
                    }
                ],
            )

            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            self.assertEqual(
                fresh_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "restart_active_behavior",
                    }
                ],
            )

    def test_training_job_restart_behavior_reconstructed_job_is_non_cancellable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            _, created_payload, _ = self._create_restart_limitation_job(
                root,
                process=process,
                log_folder="restart_read_only_behavior",
            )
            job_id = str(created_payload["id"])
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            recovered = fresh_manager.get_job_payload(job_id)
            self.assertEqual(recovered["status"], "unknown")
            with self.assertRaisesRegex(
                TrainingJobFailure,
                "live process handle|after restart",
            ):
                fresh_manager.cancel_job_payload(job_id)

            self.assertFalse(process.terminated)
            self.assertIsNone(process.exit_code)

    def test_training_job_restart_behavior_unknown_ids_remain_current_errors(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            with self.assertRaises(TrainingJobFailure) as get_context:
                manager.get_job_payload("missing")
            with self.assertRaises(TrainingJobFailure) as cancel_context:
                manager.cancel_job_payload("missing")
            test_app = create_app_with_training_service(
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                manager,
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                async with lifespan_client(
                    test_app,
                    base_url="http://localhost",
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    get_response = await client.get("/training/jobs/missing")
                    cancel_response = await client.post("/training/jobs/missing/cancel")
                    return get_response, cancel_response

            get_response, cancel_response = asyncio.run(call_api())

        self.assertEqual(str(get_context.exception), "Unknown training job 'missing'.")
        self.assertEqual(
            str(cancel_context.exception),
            "Unknown training job 'missing'.",
        )
        self.assertEqual(get_response.status_code, 400)
        self.assertEqual(cancel_response.status_code, 400)
        self.assertEqual(
            get_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )
        self.assertEqual(
            cancel_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )


if __name__ == "__main__":
    unittest.main()
