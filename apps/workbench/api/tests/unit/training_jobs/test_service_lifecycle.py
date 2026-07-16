from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs import TrainingJobFailure
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    TrainingJobServiceHarness,
)


class TrainingJobLifecycleTests(unittest.TestCase):
    def test_cancel_job_reaps_process_after_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_reap",
                monitors=[],
            )
            cancelled = manager.cancel_job_payload(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(cancelled["exitCode"], -15)

    def test_cancel_job_kills_process_that_ignores_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(ignores_terminate=True)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_kill",
                monitors=[],
            )
            cancelled = manager.cancel_job_payload(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(cancelled["exitCode"], -9)

    def test_cancel_job_preserves_completed_and_failed_terminal_states(self) -> None:
        for exit_code, expected_status in ((0, "completed"), (2, "failed")):
            with self.subTest(expected_status=expected_status):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    process = FakeProcess(exit_code=exit_code)
                    manager = TrainingJobServiceHarness(
                        root=root / "jobs",
                        logs_root=root / "logs",
                        runner=FakeRunner(process),
                    )
                    created = manager.create_job_payload(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=f"preserve_{expected_status}",
                        monitors=[],
                    )
                    job_id = str(created["id"])

                    first = manager.cancel_job_payload(job_id)
                    second = manager.cancel_job_payload(job_id)

                    self.assertEqual(first["status"], expected_status)
                    self.assertEqual(second["status"], expected_status)
                    self.assertEqual(first["exitCode"], exit_code)
                    self.assertEqual(second["exitCode"], exit_code)
                    self.assertFalse(process.terminated)
                    self.assertEqual(first["eventCounts"].get("cancelled", 0), 0)
                    self.assertEqual(second["eventCounts"].get("cancelled", 0), 0)
                    self.assertNotIn(job_id, manager.runtime._processes)

    def test_repeated_active_cancellation_emits_once_and_retains_bounded_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="idempotent_cancel",
                monitors=[],
            )
            job_id = str(created["id"])
            job = manager.jobs[job_id]

            first = manager.cancel_job_payload(job_id)
            second = manager.cancel_job_payload(job_id)

            self.assertEqual(first["status"], "cancelled")
            self.assertEqual(second["status"], "cancelled")
            self.assertEqual(first["eventCounts"]["cancelled"], 1)
            self.assertEqual(second["eventCounts"]["cancelled"], 1)
            self.assertNotIn(job_id, manager.runtime._processes)
            self.assertIn(job.progress_path, manager.runtime.progress_store._cache)
            self.assertIn(job_id, manager.runtime._live_projection_cache._cache)
            self.assertLessEqual(
                manager.runtime.progress_store.cache_stats(job).retained_event_count,
                100,
            )

    def test_observed_process_completion_retains_bounded_projection_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="terminal_cache_eviction",
                monitors=[],
            )
            job_id = str(created["id"])
            job = manager.jobs[job_id]
            self.assertIn(job_id, manager.runtime._processes)
            self.assertIn(job.progress_path, manager.runtime.progress_store._cache)
            self.assertIn(job_id, manager.runtime._live_projection_cache._cache)

            process.exit_code = 0
            completed = manager.get_job_payload(job_id)
            with patch.object(
                manager.runtime.progress_store,
                "_decode_event",
                wraps=manager.runtime.progress_store._decode_event,
            ) as decode_event:
                repeated = manager.get_job_payload(job_id)

            self.assertEqual(completed["status"], "completed")
            self.assertEqual(repeated["status"], "completed")
            decode_event.assert_not_called()
            self.assertNotIn(job_id, manager.runtime._processes)
            self.assertIn(job.progress_path, manager.runtime.progress_store._cache)
            self.assertIn(job_id, manager.runtime._live_projection_cache._cache)

    def test_terminal_transition_invalidates_its_log_experiment_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            invalidated: list[str] = []
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
                terminal_log_experiment_invalidator=invalidated.append,
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="affected_experiment",
                monitors=[],
            )
            job_id = str(created["id"])

            process.exit_code = 0
            manager.get_job_payload(job_id)
            manager.get_job_payload(job_id)

        self.assertEqual(invalidated, ["affected_experiment"])

    def test_cancel_job_failure_keeps_job_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(ignores_terminate=True, ignores_kill=True)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_failure",
                monitors=[],
            )
            job_id = str(payload["id"])

            with self.assertRaisesRegex(
                TrainingJobFailure,
                "process survived terminate and kill",
            ):
                manager.cancel_job_payload(job_id)
            current = manager.get_job_payload(job_id)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(current["status"], "running")
        self.assertIsNone(current["exitCode"])
        self.assertFalse(
            any(event.get("type") == "cancelled" for event in current["events"])
        )

    def test_terminal_progress_event_does_not_finish_live_process_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(exit_code=None)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="terminal_event_live_scope",
                monitors=[],
            )
            job = manager.jobs[str(payload["id"])]
            manager.runtime._write_event(
                job,
                {"type": "completed", "status": "completed"},
            )

            current = manager.get_job_payload(str(payload["id"]))

        self.assertEqual(current["status"], "running")
        self.assertIsNone(current["exitCode"])

    def test_training_job_get_refreshes_process_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created_payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="completed_refresh",
                monitors=[],
            )
            job = manager.jobs[str(created_payload["id"])]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "status": "running",
                },
            )

            process.exit_code = 0
            payload = manager.get_job_payload(str(created_payload["id"]))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["exitCode"], 0)
        self.assertEqual(payload["id"], created_payload["id"])
        self.assertEqual(payload["pid"], 1234)
        self.assertEqual(payload["runPlan"]["runs"][0]["status"], "Completed")
        self.assertEqual(payload["runPlan"]["summary"]["completedRuns"], 1)

    def test_training_job_get_refreshes_process_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created_payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="failed_refresh",
                monitors=[],
            )

            process.exit_code = 2
            payload = manager.get_job_payload(str(created_payload["id"]))

        failed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["exitCode"], 2)
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["error"], "Training failed")
        self.assertEqual(payload["runPlan"]["summary"]["failedRuns"], 1)


if __name__ == "__main__":
    unittest.main()
