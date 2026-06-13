"""Typed training use-case commands and views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from viewer.backend.schemas._base import ConfigValue

TrainingRunStatus = Literal[
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
]
TrainingRunChangeSource = Literal["override", "search"]


def _mapping_items(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


@dataclass(frozen=True, slots=True)
class TrainingSearch:
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]] = field(default_factory=dict)
    random_samples: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> TrainingSearch | None:
        if payload is None:
            return None
        raw_values = payload.get("values") or {}
        values = {
            str(key): list(value) if isinstance(value, list) else []
            for key, value in dict(raw_values).items()
        }
        raw_random_samples = payload.get("randomSamples")
        return cls(
            mode=cast(Literal["grid", "random"], str(payload.get("mode") or "grid")),
            values=values,
            random_samples=(
                int(raw_random_samples)
                if raw_random_samples is not None
                else None
            ),
        )

    def to_api_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "values": self.values,
        }
        if self.random_samples is not None:
            payload["randomSamples"] = self.random_samples
        return payload


@dataclass(frozen=True, slots=True)
class TrainingRunChangeView:
    key: str
    label: str
    value: ConfigValue
    source: TrainingRunChangeSource

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingRunChangeView:
        return cls(
            key=str(payload.get("key") or ""),
            label=str(payload.get("label") or ""),
            value=cast(ConfigValue, payload.get("value")),
            source=cast(
                TrainingRunChangeSource,
                str(payload.get("source") or "override"),
            ),
        )

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunView:
    id: str
    index: int
    status: TrainingRunStatus
    preset: str
    dataset: str
    changes: list[TrainingRunChangeView]
    overrides: dict[str, Any]
    command: str
    total_epochs: int
    snapshot_id: str | None = None
    snapshot_name: str | None = None
    current_epoch: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    log_dir: str | None = None
    error: str | None = None
    error_traceback: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingRunView:
        snapshot_id = payload.get("snapshotId")
        snapshot_name = payload.get("snapshotName")
        log_dir = payload.get("logDir")
        error = payload.get("error")
        error_traceback = payload.get("errorTraceback")
        return cls(
            id=str(payload.get("id") or ""),
            index=int(payload.get("index") or 0),
            status=cast(TrainingRunStatus, str(payload.get("status") or "Pending")),
            preset=str(payload.get("preset") or ""),
            snapshot_id=str(snapshot_id) if snapshot_id is not None else None,
            snapshot_name=str(snapshot_name) if snapshot_name is not None else None,
            dataset=str(payload.get("dataset") or ""),
            changes=[
                TrainingRunChangeView.from_payload(item)
                for item in _mapping_items(payload.get("changes"))
            ],
            overrides=dict(payload.get("overrides") or {}),
            command=str(payload.get("command") or ""),
            total_epochs=int(payload.get("totalEpochs") or 0),
            current_epoch=int(payload.get("currentEpoch") or 0),
            metrics=dict(payload.get("metrics") or {}),
            log_dir=str(log_dir) if log_dir is not None else None,
            error=str(error) if error is not None else None,
            error_traceback=(
                str(error_traceback) if error_traceback is not None else None
            ),
        )

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "status": self.status,
            "preset": self.preset,
            "snapshotId": self.snapshot_id,
            "snapshotName": self.snapshot_name,
            "dataset": self.dataset,
            "changes": [change.to_api_payload() for change in self.changes],
            "overrides": self.overrides,
            "command": self.command,
            "totalEpochs": self.total_epochs,
            "currentEpoch": self.current_epoch,
            "metrics": self.metrics,
            "logDir": self.log_dir,
            "error": self.error,
            "errorTraceback": self.error_traceback,
        }


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

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> TrainingRunPlanSummaryView:
        payload = payload or {}
        return cls(
            total_runs=int(payload.get("totalRuns") or 0),
            completed_runs=int(payload.get("completedRuns") or 0),
            running_runs=int(payload.get("runningRuns") or 0),
            pending_runs=int(payload.get("pendingRuns") or 0),
            failed_runs=int(payload.get("failedRuns") or 0),
            cancelled_runs=int(payload.get("cancelledRuns") or 0),
            skipped_runs=int(payload.get("skippedRuns") or 0),
            total_epochs=int(payload.get("totalEpochs") or 0),
            completed_epochs=int(payload.get("completedEpochs") or 0),
            remaining_epochs=int(payload.get("remainingEpochs") or 0),
        )

    def to_api_payload(self) -> dict[str, int]:
        return {
            "totalRuns": self.total_runs,
            "completedRuns": self.completed_runs,
            "runningRuns": self.running_runs,
            "pendingRuns": self.pending_runs,
            "failedRuns": self.failed_runs,
            "cancelledRuns": self.cancelled_runs,
            "skippedRuns": self.skipped_runs,
            "totalEpochs": self.total_epochs,
            "completedEpochs": self.completed_epochs,
            "remainingEpochs": self.remaining_epochs,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunPlanView:
    model: str
    preset: str
    presets: list[str]
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearch | None
    log_folder: str
    is_random_search: bool
    runs: list[TrainingRunView]
    summary: TrainingRunPlanSummaryView

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingRunPlanView:
        return cls(
            model=str(payload.get("model") or ""),
            preset=str(payload.get("preset") or ""),
            presets=[str(item) for item in payload.get("presets") or []],
            datasets=[str(item) for item in payload.get("datasets") or []],
            overrides=dict(payload.get("overrides") or {}),
            search=TrainingSearch.from_payload(
                cast(Mapping[str, Any] | None, payload.get("search"))
            ),
            log_folder=str(payload.get("logFolder") or ""),
            is_random_search=bool(payload.get("isRandomSearch")),
            runs=[
                TrainingRunView.from_payload(item)
                for item in _mapping_items(payload.get("runs"))
            ],
            summary=TrainingRunPlanSummaryView.from_payload(
                cast(Mapping[str, Any] | None, payload.get("summary"))
            ),
        )

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "preset": self.preset,
            "presets": self.presets,
            "datasets": self.datasets,
            "overrides": self.overrides,
            "search": self.search.to_api_payload() if self.search else None,
            "logFolder": self.log_folder,
            "isRandomSearch": self.is_random_search,
            "runs": [run.to_api_payload() for run in self.runs],
            "summary": self.summary.to_api_payload(),
        }


@dataclass(frozen=True, slots=True)
class TrainingResultLinkView:
    preset: str | None = None
    dataset: str | None = None
    log_dir: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingResultLinkView:
        preset = payload.get("preset")
        dataset = payload.get("dataset")
        log_dir = payload.get("logDir")
        return cls(
            preset=str(preset) if preset is not None else None,
            dataset=str(dataset) if dataset is not None else None,
            log_dir=str(log_dir) if log_dir is not None else None,
        )

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "dataset": self.dataset,
            "logDir": self.log_dir,
        }


@dataclass(frozen=True, slots=True)
class TrainingJobView:
    id: str
    status: str
    model: str
    preset: str
    presets: list[str]
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingJobView:
        exit_code = payload.get("exitCode")
        current_preset = payload.get("currentPreset")
        current_dataset = payload.get("currentDataset")
        epoch = payload.get("epoch")
        step = payload.get("step")
        log_dir = payload.get("logDir")
        return cls(
            id=str(payload.get("id") or ""),
            status=str(payload.get("status") or ""),
            model=str(payload.get("model") or ""),
            preset=str(payload.get("preset") or ""),
            presets=[str(item) for item in payload.get("presets") or []],
            datasets=[str(item) for item in payload.get("datasets") or []],
            overrides=dict(payload.get("overrides") or {}),
            search=TrainingSearch.from_payload(
                cast(Mapping[str, Any] | None, payload.get("search"))
            ),
            planned_run_count=int(payload.get("plannedRunCount") or 0),
            run_plan=(
                TrainingRunPlanView.from_payload(
                    cast(Mapping[str, Any], payload["runPlan"])
                )
                if payload.get("runPlan") is not None
                else None
            ),
            monitors=[str(item) for item in payload.get("monitors") or []],
            log_folder=str(payload.get("logFolder") or ""),
            created_at=str(payload.get("createdAt") or ""),
            updated_at=str(payload.get("updatedAt") or ""),
            exit_code=int(exit_code) if exit_code is not None else None,
            pid=int(payload.get("pid") or 0),
            current_preset=(
                str(current_preset) if current_preset is not None else None
            ),
            current_dataset=(
                str(current_dataset) if current_dataset is not None else None
            ),
            epoch=int(epoch) if epoch is not None else None,
            step=int(step) if step is not None else None,
            metrics=dict(payload.get("metrics") or {}),
            log_dir=str(log_dir) if log_dir is not None else None,
            events=[dict(item) for item in _mapping_items(payload.get("events"))],
            event_count=int(payload.get("eventCount") or 0),
            event_counts={
                str(key): int(value)
                for key, value in dict(payload.get("eventCounts") or {}).items()
            },
            events_truncated=bool(payload.get("eventsTruncated")),
            cluster_growth=[
                dict(item) for item in _mapping_items(payload.get("clusterGrowth"))
            ],
            log_tail=[str(item) for item in payload.get("logTail") or []],
            result_links=[
                TrainingResultLinkView.from_payload(item)
                for item in _mapping_items(payload.get("resultLinks"))
            ],
        )

    def to_api_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "model": self.model,
            "preset": self.preset,
            "presets": self.presets,
            "datasets": self.datasets,
            "overrides": self.overrides,
            "search": self.search.to_api_payload() if self.search else None,
            "plannedRunCount": self.planned_run_count,
            "runPlan": self.run_plan.to_api_payload() if self.run_plan else None,
            "monitors": self.monitors,
            "logFolder": self.log_folder,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "exitCode": self.exit_code,
            "pid": self.pid,
            "currentPreset": self.current_preset,
            "currentDataset": self.current_dataset,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": self.metrics,
            "logDir": self.log_dir,
            "events": self.events,
            "eventCount": self.event_count,
            "eventCounts": self.event_counts,
            "eventsTruncated": self.events_truncated,
            "clusterGrowth": self.cluster_growth,
            "logTail": self.log_tail,
            "resultLinks": [
                result_link.to_api_payload()
                for result_link in self.result_links
            ],
        }


@dataclass(frozen=True, slots=True)
class ActiveTrainingJob:
    id: str
    status: str
    log_folder: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ActiveTrainingJob:
        return cls(
            id=str(payload.get("id") or ""),
            status=str(payload.get("status") or ""),
            log_folder=str(payload.get("logFolder") or ""),
        )

    def to_api_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "status": self.status,
            "logFolder": self.log_folder,
        }


@dataclass(frozen=True, slots=True)
class CreateTrainingRunPlanCommand:
    model: str
    preset: str
    presets: list[str] | None
    datasets: list[str]
    overrides: dict[str, Any]
    log_folder: str
    search: TrainingSearch | None = None


@dataclass(frozen=True, slots=True)
class CreateTrainingJobCommand(CreateTrainingRunPlanCommand):
    monitors: list[str] = field(default_factory=list)
    run_plan: TrainingRunPlanView | None = None


__all__ = [
    "ActiveTrainingJob",
    "CreateTrainingJobCommand",
    "CreateTrainingRunPlanCommand",
    "TrainingJobView",
    "TrainingRunPlanView",
    "TrainingSearch",
]
