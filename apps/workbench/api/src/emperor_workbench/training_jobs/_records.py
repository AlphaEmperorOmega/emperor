from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from emperor_workbench.run_plans import (
    MaterializeTrainingRunPlanCommand,
    TrainingRunPlanView,
    TrainingSearch,
)

TrainingJobStatus = Literal[
    "queued",
    "running",
    "unknown",
    "completed",
    "failed",
    "cancelled",
]
TrainingCancellationCapability = Literal[
    "strict-cgroup",
    "process-group",
    "windows-job-object",
    "unsupported",
]
TrainingCancellationMode = Literal[
    "auto",
    "strict-cgroup",
    "process-group",
    "windows-job-object",
]
ResolvedTrainingCancellationMode = Literal[
    "strict-cgroup",
    "process-group",
    "windows-job-object",
]

DEFAULT_TRAINING_MEMORY_LIMIT_BYTES = 16 * 1024**3
DEFAULT_TRAINING_CPU_LIMIT = 8
DEFAULT_TRAINING_PROCESS_LIMIT = 512


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class TrainingResourceLimits:
    memory_bytes: int = DEFAULT_TRAINING_MEMORY_LIMIT_BYTES
    cpu_count: int = DEFAULT_TRAINING_CPU_LIMIT
    process_count: int = DEFAULT_TRAINING_PROCESS_LIMIT

    def __post_init__(self) -> None:
        for field_name, value in (
            ("memory_bytes", self.memory_bytes),
            ("cpu_count", self.cpu_count),
            ("process_count", self.process_count),
        ):
            if value < 1:
                raise ValueError(f"{field_name} must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResultLinkView:
    preset: str | None = None
    dataset: str | None = None
    log_dir: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingJobView:
    id: str
    status: str
    model: str
    preset: str
    presets: list[str]
    experiment_task: str
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearch | None
    planned_run_count: int
    run_plan: TrainingRunPlanView | None
    monitors: list[str]
    log_folder: str
    created_at: str
    updated_at: str
    exit_code: int | None
    pid: int
    cancellation_mode: str
    current_preset: str | None
    current_dataset: str | None
    epoch: int | None
    step: int | None
    metrics: dict[str, Any]
    log_dir: str | None
    events: list[dict[str, Any]]
    event_count: int
    event_counts: dict[str, int]
    events_truncated: bool
    cluster_growth: list[dict[str, Any]]
    log_tail: list[str]
    result_links: list[TrainingResultLinkView]
    log_tail_truncated: bool = False


@dataclass(frozen=True, slots=True)
class ActiveTrainingJob:
    id: str
    status: str
    log_folder: str


@dataclass(frozen=True, slots=True)
class TrainingProgressEventsPage:
    job_id: str
    offset: int
    limit: int
    total_count: int
    next_offset: int | None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CreateTrainingJobCommand:
    run_plan: MaterializeTrainingRunPlanCommand
    job_id: str | None = None


@dataclass
class TrainingJobRecord:
    id: str
    model: str
    preset: str
    presets: list[str]
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearch | None
    planned_run_count: int
    run_plan: TrainingRunPlanView
    monitors: list[str]
    log_folder: str
    observed_command: list[str]
    root: Path
    pid: int
    experiment_task: str = ""
    cancellation_mode: ResolvedTrainingCancellationMode = "process-group"
    worker_pid: int | None = None
    process_group_id: int | None = None
    cgroup_path: str | None = None
    windows_job_name: str | None = None
    status: TrainingJobStatus = "running"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    exit_code: int | None = None

    @property
    def payload_path(self) -> Path:
        return self.root / "payload.json"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def log_path(self) -> Path:
        return self.root / "training.log"


__all__ = [
    "ActiveTrainingJob",
    "CreateTrainingJobCommand",
    "TrainingCancellationCapability",
    "TrainingCancellationMode",
    "TrainingJobView",
    "TrainingJobStatus",
    "TrainingProgressEventsPage",
    "TrainingResourceLimits",
    "TrainingResultLinkView",
]
