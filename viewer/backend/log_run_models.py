"""Log Run response and deletion value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.catalog import model_identity_payload_from_id

LOG_RESPONSE_ITEM_LIMIT = 500


@dataclass(frozen=True)
class LogRun:
    id: str
    group: str | None
    experiment: str
    model: str
    preset: str
    dataset: str
    runName: str
    timestamp: str | None
    version: str
    relativePath: str
    hasResult: bool
    eventFileCount: int
    checkpointCount: int
    hasHparams: bool
    experimentTask: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    path: Path = field(repr=False, compare=False, default=Path())

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group": self.group,
            "experiment": self.experiment,
            **model_identity_payload_from_id(self.model),
            "preset": self.preset,
            "experimentTask": self.experimentTask,
            "dataset": self.dataset,
            "runName": self.runName,
            "timestamp": self.timestamp,
            "version": self.version,
            "relativePath": self.relativePath,
            "hasResult": self.hasResult,
            "eventFileCount": self.eventFileCount,
            "checkpointCount": self.checkpointCount,
            "hasHparams": self.hasHparams,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class LogCheckpoint:
    id: str
    runId: str
    filename: str
    relativePath: str
    epoch: int | None
    step: int | None
    sizeBytes: int
    modifiedAt: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "runId": self.runId,
            "filename": self.filename,
            "relativePath": self.relativePath,
            "epoch": self.epoch,
            "step": self.step,
            "sizeBytes": self.sizeBytes,
            "modifiedAt": self.modifiedAt,
        }


@dataclass(frozen=True)
class LogRunArtifact:
    id: str
    kind: str
    label: str
    relativePath: str
    sizeBytes: int
    modifiedAt: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "relativePath": self.relativePath,
            "sizeBytes": self.sizeBytes,
            "modifiedAt": self.modifiedAt,
        }


@dataclass(frozen=True)
class LogRunArtifacts:
    runId: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: list[LogRunArtifact]
    checkpoints: list[LogCheckpoint]

    def to_response(self) -> dict[str, Any]:
        return {
            "runId": self.runId,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": [artifact.to_response() for artifact in self.artifacts],
            "checkpoints": [
                checkpoint.to_response() for checkpoint in self.checkpoints
            ],
        }


@dataclass(frozen=True)
class LogExperimentDeleteResult:
    experiment: str
    deletedRunIds: list[str]
    deletedRunCount: int
    deletedRelativePath: str

    def to_response(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "deletedRunIds": self.deletedRunIds,
            "deletedRunCount": self.deletedRunCount,
            "deletedRelativePath": self.deletedRelativePath,
        }


@dataclass(frozen=True)
class LogRunDeleteFilters:
    experiments: list[str]
    datasets: list[str]
    models: list[str]
    presets: list[str]
    runIds: list[str]


@dataclass(frozen=True)
class LogRunDeleteCandidate:
    id: str
    experiment: str
    model: str
    preset: str
    dataset: str
    runName: str
    version: str
    relativePath: str
    path: Path = field(repr=False, compare=False, default=Path())

    @classmethod
    def from_run(cls, run: LogRun) -> LogRunDeleteCandidate:
        return cls(
            id=run.id,
            experiment=run.experiment,
            model=run.model,
            preset=run.preset,
            dataset=run.dataset,
            runName=run.runName,
            version=run.version,
            relativePath=run.relativePath,
            path=run.path,
        )

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment": self.experiment,
            **model_identity_payload_from_id(self.model),
            "preset": self.preset,
            "dataset": self.dataset,
            "runName": self.runName,
            "version": self.version,
            "relativePath": self.relativePath,
        }


@dataclass(frozen=True)
class ActiveLogRunDeleteBlocker:
    id: str
    logFolder: str
    status: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "logFolder": self.logFolder,
            "status": self.status,
        }


@dataclass(frozen=True)
class LogRunDeletePlan:
    candidates: list[LogRunDeleteCandidate]
    blockedByActiveJobs: list[ActiveLogRunDeleteBlocker] = field(default_factory=list)

    @property
    def canDelete(self) -> bool:
        return bool(self.candidates) and not self.blockedByActiveJobs

    def to_response(self) -> dict[str, Any]:
        return _delete_plan_response_fields(
            self.candidates,
            blocked_by_active_jobs=self.blockedByActiveJobs,
            can_delete=self.canDelete,
        )


@dataclass(frozen=True)
class LogRunDeleteResult:
    candidates: list[LogRunDeleteCandidate]
    deletedRunIds: list[str]
    deletedRelativePaths: list[str]

    def to_response(self) -> dict[str, Any]:
        return {
            "deletedRunIds": self.deletedRunIds,
            "deletedRunCount": len(self.deletedRunIds),
            "deletedRelativePaths": self.deletedRelativePaths,
            **_delete_plan_response_fields(
                self.candidates,
                blocked_by_active_jobs=[],
                can_delete=True,
            ),
        }


@dataclass(frozen=True)
class LogExperiment:
    experiment: str
    runCount: int
    relativePath: str

    def to_response(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "runCount": self.runCount,
            "relativePath": self.relativePath,
        }


def _delete_plan_response_fields(
    candidates: list[LogRunDeleteCandidate],
    *,
    blocked_by_active_jobs: list[ActiveLogRunDeleteBlocker],
    can_delete: bool,
) -> dict[str, Any]:
    affected = _affected_values(candidates)
    returned_candidates = candidates[:LOG_RESPONSE_ITEM_LIMIT]
    truncated = len(candidates) > len(returned_candidates)
    return {
        "candidateCount": len(candidates),
        "sourceItemCount": len(candidates),
        "returnedItemCount": len(returned_candidates),
        "truncated": truncated,
        "truncationReason": (
            f"delete candidates capped at {LOG_RESPONSE_ITEM_LIMIT} rows"
            if truncated
            else None
        ),
        "counts": {
            "runs": len(candidates),
            "experiments": len(affected["experiments"]),
            "datasets": len(affected["datasets"]),
            "models": len(affected["models"]),
            "presets": len(affected["presets"]),
        },
        "affected": affected,
        "candidates": [candidate.to_response() for candidate in returned_candidates],
        "blockedByActiveJobs": [
            blocker.to_response() for blocker in blocked_by_active_jobs
        ],
        "canDelete": can_delete,
    }


def _affected_values(
    candidates: list[LogRunDeleteCandidate],
) -> dict[str, Any]:
    model_ids = sorted({candidate.model for candidate in candidates})
    return {
        "experiments": sorted({candidate.experiment for candidate in candidates}),
        "datasets": sorted({candidate.dataset for candidate in candidates}),
        "models": [
            model_identity_payload_from_id(model_id) for model_id in model_ids
        ],
        "presets": sorted({candidate.preset for candidate in candidates}),
        "runIds": sorted({candidate.id for candidate in candidates}),
    }


__all__ = [
    "LOG_RESPONSE_ITEM_LIMIT",
    "ActiveLogRunDeleteBlocker",
    "LogCheckpoint",
    "LogExperiment",
    "LogExperimentDeleteResult",
    "LogRun",
    "LogRunArtifact",
    "LogRunArtifacts",
    "LogRunDeleteCandidate",
    "LogRunDeleteFilters",
    "LogRunDeletePlan",
    "LogRunDeleteResult",
]
