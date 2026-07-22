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
class TrainingCommandsView:
    posix: str
    powershell: str


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
    total_epochs: int
    command_argv: list[str] = field(default_factory=list)
    commands: TrainingCommandsView = field(
        default_factory=lambda: TrainingCommandsView(posix="", powershell="")
    )
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
    """Authoritative Run choices accepted from an untrusted caller."""

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
class MaterializeTrainingRunPlanCommand(CreateTrainingRunPlanCommand):
    submitted_plan: SubmittedTrainingRunPlan | None = None
    snapshot_revisions: tuple[ConfigSnapshotRevision, ...] = ()


@dataclass(frozen=True, slots=True)
class MaterializedTrainingRunPlan:
    plan: TrainingRunPlanView
    monitors: tuple[str, ...]


__all__ = [
    "ConfigSnapshotRevision",
    "ConfigValue",
    "CreateTrainingRunPlanCommand",
    "MaterializeTrainingRunPlanCommand",
    "MaterializedTrainingRunPlan",
    "SubmittedTrainingRun",
    "SubmittedTrainingRunPlan",
    "TrainingCommandsView",
    "TrainingRunChangeSource",
    "TrainingRunChangeView",
    "TrainingRunPlanSummaryView",
    "TrainingRunPlanView",
    "TrainingRunStatus",
    "TrainingRunView",
    "TrainingSearch",
]
