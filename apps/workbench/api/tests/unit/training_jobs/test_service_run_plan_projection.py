from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    TrainingJobServiceHarness,
)


class TrainingJobRunPlanProjectionTests(unittest.TestCase):
    def _create_progress_projection_job(
        self,
        root: Path,
        *,
        process: FakeProcess | None = None,
        run_count: int = 3,
        total_epochs: int = 4,
    ):
        datasets = ["Mnist", "Cifar10", "Cifar100"][:run_count]
        run_plan = {
            "runs": [
                {
                    "id": f"{dataset.lower()}-row",
                    "preset": "baseline",
                    "dataset": dataset,
                    "overrides": {"NUM_EPOCHS": total_epochs},
                }
                for dataset in datasets
            ]
        }
        manager = TrainingJobServiceHarness(
            root=root / "jobs",
            logs_root=root / "logs",
            runner=FakeRunner(process),
        )
        payload = manager.create_job_payload(
            model="linears/linear",
            preset="baseline",
            datasets=datasets,
            overrides={},
            log_folder="progress_projection",
            run_plan=run_plan,
        )
        return manager, payload, manager.jobs[str(payload["id"])]

    def test_training_run_plan_preserves_error_traceback_from_progress_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="traceback_test",
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "error",
                    "status": "failed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": payload["runPlan"]["runs"][0]["id"],
                    "error": "scalar conversion failed",
                    "traceback": (
                        "Traceback (most recent call last):\n"
                        "RuntimeError: scalar conversion failed"
                    ),
                },
            )

            failed_payload = manager.get_job_payload(payload["id"])

        failed_run = failed_payload["runPlan"]["runs"][0]
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["error"], "scalar conversion failed")
        self.assertIn("RuntimeError", failed_run["errorTraceback"])

    def test_training_run_progress_projection_tracks_running_epoch_metrics_and_log_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager.runtime._write_event(
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

            started_payload = manager.get_job_payload(str(created_payload["id"]))

            manager.runtime._write_event(
                job,
                {
                    "type": "epoch_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 3,
                    "logDir": log_dir,
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 2,
                    "step": 7,
                    "metrics": {"train/loss": 0.4},
                    "logDir": log_dir,
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "validation",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 0,
                    "step": 8,
                    "metrics": {"validation/accuracy": 0.75},
                    "logDir": log_dir,
                },
            )

            progressed_payload = manager.get_job_payload(str(created_payload["id"]))

        started_run = started_payload["runPlan"]["runs"][0]
        self.assertEqual(started_run["status"], "Running")
        self.assertEqual(started_run["currentEpoch"], 0)
        self.assertEqual(started_run["logDir"], log_dir)
        self.assertEqual(started_payload["runPlan"]["summary"]["runningRuns"], 1)
        self.assertEqual(started_payload["runPlan"]["summary"]["pendingRuns"], 2)

        progressed_run = progressed_payload["runPlan"]["runs"][0]
        self.assertEqual(progressed_run["status"], "Running")
        self.assertEqual(progressed_run["currentEpoch"], 3)
        self.assertEqual(progressed_run["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(progressed_run["logDir"], log_dir)
        self.assertEqual(progressed_payload["currentPreset"], "baseline")
        self.assertEqual(progressed_payload["currentDataset"], "Mnist")
        self.assertEqual(progressed_payload["step"], 8)
        self.assertEqual(progressed_payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(progressed_payload["logDir"], log_dir)
        self.assertEqual(
            progressed_payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 1,
                "pendingRuns": 2,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 12,
                "completedEpochs": 3,
                "remainingEpochs": 9,
            },
        )

    def test_training_run_progress_projection_completes_run_and_updates_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.82},
                    "logDir": log_dir,
                },
            )

            payload = manager.get_job_payload(str(created_payload["id"]))

        completed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(completed_run["status"], "Completed")
        self.assertEqual(completed_run["currentEpoch"], 4)
        self.assertEqual(completed_run["metrics"], {"validation/accuracy": 0.82})
        self.assertEqual(completed_run["logDir"], log_dir)
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Completed", "Pending", "Pending"],
        )
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 1,
                "runningRuns": 0,
                "pendingRuns": 2,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 12,
                "completedEpochs": 4,
                "remainingEpochs": 8,
            },
        )
        self.assertEqual(
            payload["resultLinks"],
            [
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": log_dir,
                }
            ],
        )

    def test_failed_event_preserves_traceback_and_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=FakeProcess(exit_code=1),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager.runtime._write_event(
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
            manager.runtime._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 12,
                    "metrics": {"train/loss": 0.6},
                    "logDir": log_dir,
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "error",
                    "status": "failed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 2,
                    "step": 13,
                    "error": "optimizer exploded",
                    "traceback": (
                        "Traceback (most recent call last):\n"
                        "RuntimeError: optimizer exploded"
                    ),
                    "logDir": log_dir,
                },
            )

            payload = manager.get_job_payload(str(created_payload["id"]))

        failed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["currentEpoch"], 3)
        self.assertEqual(failed_run["error"], "optimizer exploded")
        self.assertIn("RuntimeError", failed_run["errorTraceback"])
        self.assertEqual(failed_run["metrics"], {"train/loss": 0.6})
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Failed", "Skipped", "Skipped"],
        )
        self.assertEqual(payload["step"], 13)
        self.assertEqual(payload["metrics"], {"train/loss": 0.6})
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 1,
                "cancelledRuns": 0,
                "skippedRuns": 2,
                "totalEpochs": 12,
                "completedEpochs": 3,
                "remainingEpochs": 0,
            },
        )

    def test_failed_process_marks_running_and_pending_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=process,
            )
            runs = created_payload["runPlan"]["runs"]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": runs[0]["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.82},
                    "logDir": (
                        "logs/progress_projection/linear/baseline/Mnist/version_0"
                    ),
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Cifar10",
                    "preset": "baseline",
                    "runId": runs[1]["id"],
                    "runIndex": 2,
                    "logDir": (
                        "logs/progress_projection/linear/baseline/Cifar10/version_0"
                    ),
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Cifar10",
                    "preset": "baseline",
                    "runId": runs[1]["id"],
                    "runIndex": 2,
                    "epoch": 1,
                    "step": 9,
                    "metrics": {"train/loss": 0.7},
                },
            )
            process.exit_code = 2

            payload = manager.get_job_payload(str(created_payload["id"]))

        projected_runs = payload["runPlan"]["runs"]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["exitCode"], 2)
        self.assertEqual(
            [run["status"] for run in projected_runs],
            ["Completed", "Failed", "Skipped"],
        )
        self.assertEqual(projected_runs[1]["currentEpoch"], 2)
        self.assertIsNone(projected_runs[1]["error"])
        self.assertIsNone(projected_runs[2]["error"])
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 1,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 1,
                "cancelledRuns": 0,
                "skippedRuns": 1,
                "totalEpochs": 12,
                "completedEpochs": 6,
                "remainingEpochs": 0,
            },
        )

    def test_cancelled_job_marks_running_and_pending_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=process,
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager.runtime._write_event(
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
            manager.runtime._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 6,
                    "metrics": {"train/loss": 0.9},
                    "logDir": log_dir,
                },
            )

            manager.cancel_job_payload(str(created_payload["id"]))
            payload = manager.get_job_payload(str(created_payload["id"]))

        projected_runs = payload["runPlan"]["runs"]
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(payload["events"][-1]["type"], "cancelled")
        self.assertEqual(
            [run["status"] for run in projected_runs],
            ["Cancelled", "Skipped", "Skipped"],
        )
        self.assertEqual(projected_runs[0]["currentEpoch"], 2)
        self.assertEqual(projected_runs[0]["metrics"], {"train/loss": 0.9})
        self.assertEqual(projected_runs[0]["logDir"], log_dir)
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 0,
                "cancelledRuns": 1,
                "skippedRuns": 2,
                "totalEpochs": 12,
                "completedEpochs": 2,
                "remainingEpochs": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
