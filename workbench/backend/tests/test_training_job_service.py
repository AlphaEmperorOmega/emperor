from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import LogExperimentMutationCoordinator
from workbench.backend.tests.helpers import FakeProcess, FakeRunner
from workbench.backend.training_jobs.contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingJobView,
)
from workbench.backend.training_jobs.service import TrainingJobService


def _command(log_folder: str) -> CreateTrainingJobCommand:
    return CreateTrainingJobCommand(
        model="linears/linear",
        preset="baseline",
        presets=None,
        datasets=["Mnist"],
        overrides={},
        log_folder=log_folder,
        monitors=[],
    )


def _service(
    root: Path,
    *,
    process: FakeProcess | None = None,
) -> TrainingJobService:
    return TrainingJobService(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=FakeRunner(process),
        mutation_coordinator=LogExperimentMutationCoordinator(),
    )


class TrainingJobServiceWorkflowTests(unittest.TestCase):
    def test_create_read_cancel_and_active_workflow_uses_public_interface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            service = _service(Path(tmp), process=process)

            created = service.create_job(_command("public_workflow"))
            observed = service.get_job(created.id)
            active = service.active_jobs()
            cancelled = service.cancel_job(created.id)

        self.assertIsInstance(created, TrainingJobView)
        self.assertEqual(observed.id, created.id)
        self.assertEqual(observed.status, "running")
        self.assertEqual(
            active,
            [
                ActiveTrainingJob(
                    id=created.id,
                    status="running",
                    log_folder="public_workflow",
                )
            ],
        )
        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(service.active_jobs(), [])

    def test_fresh_service_recovers_unknown_job_through_public_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = _service(root).create_job(_command("public_restart"))

            restarted = _service(root)
            recovered = restarted.get_job(created.id)
            active = restarted.active_jobs()
            with self.assertRaisesRegex(InspectorError, "live process handle"):
                restarted.cancel_job(created.id)

        self.assertIsInstance(recovered, TrainingJobView)
        self.assertEqual(recovered.status, "unknown")
        self.assertEqual(
            active,
            [
                ActiveTrainingJob(
                    id=created.id,
                    status="unknown",
                    log_folder="public_restart",
                )
            ],
        )

    def test_fresh_service_recovers_terminal_progress_through_public_interface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = _service(root).create_job(_command("public_recovery"))
            assert created.run_plan is not None
            run = created.run_plan.runs[0]
            progress_path = root / "jobs" / created.id / "progress.jsonl"
            with progress_path.open("a", encoding="utf-8") as progress_file:
                progress_file.write(
                    json.dumps(
                        {
                            "type": "dataset_completed",
                            "status": "running",
                            "dataset": run.dataset,
                            "preset": run.preset,
                            "runId": run.id,
                            "runIndex": 1,
                        }
                    )
                    + "\n"
                )
                progress_file.write(
                    json.dumps(
                        {
                            "type": "completed",
                            "status": "completed",
                            "jobId": created.id,
                        }
                    )
                    + "\n"
                )

            restarted = _service(root)
            recovered = restarted.get_job(created.id)

        self.assertIsInstance(recovered, TrainingJobView)
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.exit_code, 0)
        self.assertEqual(recovered.event_counts["completed"], 1)
        assert recovered.run_plan is not None
        self.assertEqual(recovered.run_plan.summary.completed_runs, 1)
        self.assertEqual(restarted.active_jobs(), [])


if __name__ == "__main__":
    unittest.main()
