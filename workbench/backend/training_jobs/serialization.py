"""Explicit compatibility serialization for Training Job and Run-plan values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from emperor.model_packages import (
    model_id_from_payload,
    model_identity_payload_from_id,
)

from workbench.backend.training_jobs.contracts import (
    TrainingRunChangeSource,
    TrainingRunChangeView,
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
    TrainingRunStatus,
    TrainingRunView,
    TrainingSearch,
)


def _mapping_items(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _payload_model_id(payload: Mapping[str, Any]) -> str:
    model_id = model_id_from_payload(payload)
    if model_id is not None:
        return model_id
    return str(payload.get("model") or "")


def training_search_from_payload(
    payload: Mapping[str, Any] | None,
) -> TrainingSearch | None:
    if payload is None:
        return None
    raw_values = payload.get("values") or {}
    values = {
        str(key): list(value) if isinstance(value, list) else []
        for key, value in dict(raw_values).items()
    }
    raw_random_samples = payload.get("randomSamples")
    return TrainingSearch(
        mode=cast(
            Literal["grid", "random"],
            str(payload.get("mode") or "grid"),
        ),
        values=values,
        random_samples=(
            int(raw_random_samples) if raw_random_samples is not None else None
        ),
    )


def training_search_to_payload(search: TrainingSearch) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": search.mode, "values": search.values}
    if search.random_samples is not None:
        payload["randomSamples"] = search.random_samples
    return payload


def training_run_change_from_payload(
    payload: Mapping[str, Any],
) -> TrainingRunChangeView:
    return TrainingRunChangeView(
        key=str(payload.get("key") or ""),
        label=str(payload.get("label") or ""),
        value=payload.get("value"),
        source=cast(
            TrainingRunChangeSource,
            str(payload.get("source") or "override"),
        ),
    )


def training_run_change_to_payload(change: TrainingRunChangeView) -> dict[str, Any]:
    return {
        "key": change.key,
        "label": change.label,
        "value": change.value,
        "source": change.source,
    }


def training_run_from_payload(payload: Mapping[str, Any]) -> TrainingRunView:
    snapshot_id = payload.get("snapshotId")
    snapshot_name = payload.get("snapshotName")
    log_dir = payload.get("logDir")
    error = payload.get("error")
    error_traceback = payload.get("errorTraceback")
    return TrainingRunView(
        id=str(payload.get("id") or ""),
        index=int(payload.get("index") or 0),
        status=cast(TrainingRunStatus, str(payload.get("status") or "Pending")),
        preset=str(payload.get("preset") or ""),
        dataset=str(payload.get("dataset") or ""),
        experiment_task=str(payload.get("experimentTask") or ""),
        changes=[
            training_run_change_from_payload(item)
            for item in _mapping_items(payload.get("changes"))
        ],
        overrides=dict(payload.get("overrides") or {}),
        command=str(payload.get("command") or ""),
        total_epochs=int(payload.get("totalEpochs") or 0),
        snapshot_id=str(snapshot_id) if snapshot_id is not None else None,
        snapshot_name=str(snapshot_name) if snapshot_name is not None else None,
        snapshot_id_present="snapshotId" in payload,
        snapshot_name_present="snapshotName" in payload,
        current_epoch=int(payload.get("currentEpoch") or 0),
        metrics=dict(payload.get("metrics") or {}),
        log_dir=str(log_dir) if log_dir is not None else None,
        error=str(error) if error is not None else None,
        error_traceback=(
            str(error_traceback) if error_traceback is not None else None
        ),
    )


def training_run_to_payload(run: TrainingRunView) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": run.id,
        "index": run.index,
        "status": run.status,
        "preset": run.preset,
        "dataset": run.dataset,
        "experimentTask": run.experiment_task,
        "changes": [
            training_run_change_to_payload(change)
            for change in run.changes
        ],
        "overrides": run.overrides,
        "command": run.command,
        "totalEpochs": run.total_epochs,
        "currentEpoch": run.current_epoch,
        "metrics": run.metrics,
        "logDir": run.log_dir,
        "error": run.error,
        "errorTraceback": run.error_traceback,
    }
    if run.snapshot_id_present or run.snapshot_id is not None:
        payload["snapshotId"] = run.snapshot_id
    if run.snapshot_name_present or run.snapshot_name is not None:
        payload["snapshotName"] = run.snapshot_name
    return payload


def training_summary_from_payload(
    payload: Mapping[str, Any] | None,
) -> TrainingRunPlanSummaryView:
    payload = payload or {}
    return TrainingRunPlanSummaryView(
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


def training_summary_to_payload(summary: TrainingRunPlanSummaryView) -> dict[str, int]:
    return {
        "totalRuns": summary.total_runs,
        "completedRuns": summary.completed_runs,
        "runningRuns": summary.running_runs,
        "pendingRuns": summary.pending_runs,
        "failedRuns": summary.failed_runs,
        "cancelledRuns": summary.cancelled_runs,
        "skippedRuns": summary.skipped_runs,
        "totalEpochs": summary.total_epochs,
        "completedEpochs": summary.completed_epochs,
        "remainingEpochs": summary.remaining_epochs,
    }


def training_run_plan_from_payload(payload: Mapping[str, Any]) -> TrainingRunPlanView:
    return TrainingRunPlanView(
        model=_payload_model_id(payload),
        preset=str(payload.get("preset") or ""),
        presets=[str(item) for item in payload.get("presets") or []],
        experiment_task=str(payload.get("experimentTask") or ""),
        datasets=[str(item) for item in payload.get("datasets") or []],
        overrides=dict(payload.get("overrides") or {}),
        search=training_search_from_payload(
            cast(Mapping[str, Any] | None, payload.get("search"))
        ),
        log_folder=str(payload.get("logFolder") or ""),
        is_random_search=bool(payload.get("isRandomSearch")),
        runs=[
            training_run_from_payload(item)
            for item in _mapping_items(payload.get("runs"))
        ],
        summary=training_summary_from_payload(
            cast(Mapping[str, Any] | None, payload.get("summary"))
        ),
    )


def training_run_plan_to_payload(plan: TrainingRunPlanView) -> dict[str, Any]:
    return {
        **model_identity_payload_from_id(plan.model),
        "preset": plan.preset,
        "presets": plan.presets,
        "experimentTask": plan.experiment_task,
        "datasets": plan.datasets,
        "overrides": plan.overrides,
        "search": training_search_to_payload(plan.search) if plan.search else None,
        "logFolder": plan.log_folder,
        "isRandomSearch": plan.is_random_search,
        "runs": [training_run_to_payload(run) for run in plan.runs],
        "summary": training_summary_to_payload(plan.summary),
    }


__all__ = [
    "training_run_plan_from_payload",
    "training_run_plan_to_payload",
    "training_search_from_payload",
    "training_search_to_payload",
]
