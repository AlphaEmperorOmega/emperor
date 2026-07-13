"""Transport-neutral Training Job commands and frozen caller snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ConfigValue = bool | int | float | str | None
TrainingRunStatus = Literal[
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
]
TrainingRunChangeSource = Literal["override", "search"]


@dataclass(frozen=True, slots=True)
class ConfigSnapshotRevision:
    id: str
    semantic_revision: str


@dataclass(frozen=True, slots=True)
class TrainingSearch:
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]] = field(default_factory=dict)
    random_samples: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunChangeView:
    key: str
    label: str
    value: ConfigValue
    source: TrainingRunChangeSource


@dataclass(frozen=True, slots=True)
class TrainingRunView:
    id: str
    index: int
    status: TrainingRunStatus
    preset: str
    dataset: str
    experiment_task: str
    changes: list[TrainingRunChangeView]
    overrides: dict[str, Any]
    command: str
    total_epochs: int
    snapshot_id: str | None = None
    snapshot_name: str | None = None
    snapshot_id_present: bool = False
    snapshot_name_present: bool = False
    current_epoch: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    log_dir: str | None = None
    error: str | None = None
    error_traceback: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunPlanSummaryView:
    total_runs: int = 0
    completed_runs: int = 0
    running_runs: int = 0
    pending_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    skipped_runs: int = 0
    total_epochs: int = 0
    completed_epochs: int = 0
    remaining_epochs: int = 0


@dataclass(frozen=True, slots=True)
class TrainingRunPlanView:
    model: str
    preset: str
    presets: list[str]
    experiment_task: str
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearch | None
    log_folder: str
    is_random_search: bool
    runs: list[TrainingRunView]
    summary: TrainingRunPlanSummaryView
    snapshot_revisions: tuple[ConfigSnapshotRevision, ...] = ()


@dataclass(frozen=True, slots=True)
class SubmittedTrainingRun:
    """Authoritative row choices accepted from an untrusted caller."""

    id: str
    preset: str
    dataset: str
    overrides: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str | None = None
    snapshot_name: str | None = None


@dataclass(frozen=True, slots=True)
class SubmittedTrainingRunPlan:
    """Minimal submitted Run Plan; all presentation state is derived."""

    runs: list[SubmittedTrainingRun] = field(default_factory=list)
    snapshot_revisions: tuple[ConfigSnapshotRevision, ...] = ()


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
class CreateTrainingRunPlanCommand:
    model: str
    preset: str
    presets: list[str] | None
    datasets: list[str]
    overrides: dict[str, Any]
    log_folder: str
    experiment_task: str | None = None
    monitors: list[str] = field(default_factory=list)
    search: TrainingSearch | None = None
    snapshot_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CreateTrainingJobCommand(CreateTrainingRunPlanCommand):
    run_plan: SubmittedTrainingRunPlan | None = None
    snapshot_revisions: tuple[ConfigSnapshotRevision, ...] = ()


__all__ = [
    "ActiveTrainingJob",
    "ConfigSnapshotRevision",
    "ConfigValue",
    "CreateTrainingJobCommand",
    "CreateTrainingRunPlanCommand",
    "SubmittedTrainingRun",
    "SubmittedTrainingRunPlan",
    "TrainingJobView",
    "TrainingProgressEventsPage",
    "TrainingRunChangeSource",
    "TrainingRunChangeView",
    "TrainingRunPlanSummaryView",
    "TrainingRunPlanView",
    "TrainingRunStatus",
    "TrainingRunView",
    "TrainingSearch",
]
