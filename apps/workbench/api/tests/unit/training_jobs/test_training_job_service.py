from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emperor_workbench.failures import FailureKind
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.run_plans import (
    MaterializeTrainingRunPlanCommand,
    RunPlanFailure,
    RunPlanService,
)
from emperor_workbench.training_jobs import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    TrainingJobFailure,
    TrainingJobService,
    TrainingJobView,
)
from tests.support.model_packages import project_adapter_client
from tests.support.training_jobs import FakeProcess, FakeRunner


def _command(log_folder: str) -> CreateTrainingJobCommand:
    return CreateTrainingJobCommand(
        run_plan=MaterializeTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=None,
            datasets=["Mnist"],
            overrides={},
            log_folder=log_folder,
            monitors=[],
        )
    )


def _service(
    root: Path,
    *,
    process: FakeProcess | None = None,
) -> TrainingJobService:
    project_adapter = project_adapter_client()
    return TrainingJobService(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=FakeRunner(process),
        mutation_coordinator=LogExperimentMutationCoordinator(),
        run_plans=RunPlanService(
            model_packages=ModelPackageCatalog(project_adapter),
        ),
    )


class TrainingJobServiceWorkflowTests(unittest.TestCase):
    def test_run_plan_failures_are_translated_at_the_training_job_seam(self) -> None:
        class RejectingRunPlans:
            def materialize(self, *_args, **_kwargs):
                raise RunPlanFailure(
                    "stale run plan",
                    kind=FailureKind.CONFLICT,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TrainingJobService(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
                mutation_coordinator=LogExperimentMutationCoordinator(),
                run_plans=RejectingRunPlans(),
            )

            with self.assertRaises(TrainingJobFailure) as raised:
                service.create_job(_command("translated_failure"))

        self.assertEqual(raised.exception.detail, "stale run plan")
        self.assertEqual(raised.exception.kind, FailureKind.CONFLICT)

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
            with self.assertRaisesRegex(
                TrainingJobFailure,
                "live process handle",
            ):
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
