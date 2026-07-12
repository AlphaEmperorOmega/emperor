"""Transport-neutral Training Job commands and frozen caller snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workbench.backend.training_jobs.run_plan_adapter import (
    SubmittedTrainingRunPlan,
    TrainingRunPlanView,
    TrainingSearch,
)


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


@dataclass(frozen=True, slots=True)
class CreateTrainingJobCommand(CreateTrainingRunPlanCommand):
    run_plan: SubmittedTrainingRunPlan | None = None


__all__ = [
    "ActiveTrainingJob",
    "CreateTrainingJobCommand",
    "CreateTrainingRunPlanCommand",
    "TrainingJobView",
    "TrainingProgressEventsPage",
    "TrainingRunPlanView",
    "TrainingSearch",
]
