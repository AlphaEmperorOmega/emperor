from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.run_plans import (
    RunPlanPersistenceCodec,
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
)
from emperor_workbench.training_jobs._memory_store import InMemoryTrainingJobStore
from emperor_workbench.training_jobs._records import TrainingJobRecord
from tests.support.training_jobs import (
    FakeProcess as ServiceFakeProcess,
)
from tests.support.training_jobs import (
    FakeRunner as ServiceFakeRunner,
)
from tests.support.training_jobs import (
    TrainingJobServiceHarness as PublicTrainingJobServiceHarness,
)


class TrainingJobServiceHarness(PublicTrainingJobServiceHarness):
    """Owner-local white-box access for Training Jobs unit tests only."""

    @property
    def runtime(self):
        return self.service._runtime

    @property
    def jobs(self) -> dict[str, TrainingJobRecord]:
        return {job.id: job for job in self.runtime.job_store.list()}


class FailingTrainingJobStore(InMemoryTrainingJobStore):
    def save(self, job) -> None:
        raise RuntimeError("job store failed")


class FakeCgroup:
    cgroup_path = "/sys/fs/cgroup/emperor-workbench-training/job-test"

    def __init__(self, *, ignores_terminate: bool = False) -> None:
        self.processes = True
        self.terminated = False
        self.killed = False
        self.cleaned = False
        self.ignores_terminate = ignores_terminate

    def has_processes(self) -> bool:
        return self.processes

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_terminate:
            self.processes = False

    def kill(self) -> None:
        self.killed = True
        self.processes = False

    def wait_empty(self, timeout: float | None = None) -> None:
        if self.processes:
            raise TimeoutError("fake cgroup still has processes")

    def cleanup_empty(self) -> None:
        if not self.processes:
            self.cleaned = True


class FakeCgroupManager:
    def __init__(self, cgroup: FakeCgroup | None = None) -> None:
        self.cgroup = cgroup or FakeCgroup()
        self.requested_job_ids: list[str] = []

    def from_job_id(self, job_id: str):
        self.requested_job_ids.append(job_id)
        return self.cgroup


def create_progress_test_job(
    root: Path,
) -> tuple[TrainingJobServiceHarness, dict[str, object], Path]:
    manager = TrainingJobServiceHarness(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=ServiceFakeRunner(),
    )
    payload = manager.create_job_payload(
        model="linears/linear",
        preset="baseline",
        datasets=["Mnist"],
        overrides={},
        log_folder="progress_jsonl",
        monitors=[],
    )
    return manager, payload, manager.progress_path(str(payload["id"]))


def create_restart_limitation_job(
    root: Path,
    *,
    process: ServiceFakeProcess | None = None,
    log_folder: str = "restart_limitation",
) -> tuple[TrainingJobServiceHarness, dict[str, object], ServiceFakeProcess]:
    process = process or ServiceFakeProcess()
    manager = TrainingJobServiceHarness(
        root=root / "jobs",
        logs_root=root / "logs",
        runner=ServiceFakeRunner(process),
    )
    payload = manager.create_job_payload(
        model="linears/linear",
        preset="baseline",
        datasets=["Mnist"],
        overrides={},
        log_folder=log_folder,
        monitors=[],
    )
    return manager, payload, process


class FakeProcess:
    pid = 123

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return -15

    def kill(self) -> None:
        return None


class RecordingRunner:
    def __init__(self) -> None:
        self.process = FakeProcess()
        self.calls: list[dict[str, object]] = []

    def start(self, command, *, cwd, env, log_path):
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "log_path": log_path,
            }
        )
        Path(log_path).write_text("worker log\n", encoding="utf-8")
        return self.process


def make_job(root: Path) -> TrainingJobRecord:
    return TrainingJobRecord(
        id="job-1",
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        datasets=["Mnist"],
        overrides={},
        search=None,
        planned_run_count=1,
        run_plan=RunPlanPersistenceCodec.decode(
            {
                "model": "linears/linear",
                "preset": "baseline",
                "presets": ["baseline"],
                "datasets": ["Mnist"],
                "overrides": {},
                "search": None,
                "logFolder": "collaborator_test",
                "isRandomSearch": False,
                "runs": [
                    {
                        "id": "run-1",
                        "index": 1,
                        "status": "Pending",
                        "preset": "baseline",
                        "dataset": "Mnist",
                        "changes": [],
                        "overrides": {},
                        "command": "train",
                        "totalEpochs": 2,
                        "currentEpoch": 0,
                        "metrics": {},
                        "logDir": None,
                        "error": None,
                        "errorTraceback": None,
                    }
                ],
                "summary": {
                    "totalRuns": 1,
                    "pendingRuns": 1,
                    "totalEpochs": 2,
                    "remainingEpochs": 2,
                },
            }
        ),
        monitors=["linear"],
        log_folder="collaborator_test",
        observed_command=["train"],
        root=root,
        pid=123,
    )


def make_record(job_id: str = "job-1") -> TrainingJobRecord:
    return TrainingJobRecord(
        id=job_id,
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        datasets=["Mnist"],
        overrides={},
        search=None,
        planned_run_count=1,
        run_plan=TrainingRunPlanView(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            experiment_task="",
            datasets=["Mnist"],
            overrides={},
            search=None,
            log_folder="test_model",
            is_random_search=False,
            runs=[],
            summary=TrainingRunPlanSummaryView(),
        ),
        monitors=[],
        log_folder="test_model",
        observed_command=[
            "python",
            "-m",
            "emperor_workbench.training_jobs.worker",
        ],
        root=Path("/tmp/emperor-workbench-training") / job_id,
        pid=1234,
    )


__all__ = [
    "FailingTrainingJobStore",
    "FakeCgroup",
    "FakeCgroupManager",
    "FakeProcess",
    "RecordingRunner",
    "TrainingJobServiceHarness",
    "create_progress_test_job",
    "create_restart_limitation_job",
    "make_job",
    "make_record",
]
