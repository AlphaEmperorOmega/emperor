from __future__ import annotations

from typing import Any

from emperor_workbench.api.v1.run_plans._contracts import (
    SubmittedTrainingRunPlanRequest,
    TrainingSearchRequest,
)
from emperor_workbench.model_packages import ModelPackageIdentity
from emperor_workbench.run_plans import (
    ConfigSnapshotRevision,
    SubmittedTrainingRun,
    SubmittedTrainingRunPlan,
    TrainingRunChangeView,
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
    TrainingRunView,
    TrainingSearch,
)


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def search_from_request(request: TrainingSearchRequest | None) -> TrainingSearch | None:
    if request is None:
        return None
    return TrainingSearch(
        mode=request.mode,
        values={key: list(values) for key, values in request.values.items()},
        random_samples=request.randomSamples,
    )


def search_to_payload(search: TrainingSearch) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": search.mode, "values": search.values}
    if search.random_samples is not None:
        payload["randomSamples"] = search.random_samples
    return payload


def submitted_plan_from_request(
    request: SubmittedTrainingRunPlanRequest,
) -> SubmittedTrainingRunPlan:
    return SubmittedTrainingRunPlan(
        runs=[
            SubmittedTrainingRun(
                id=row.id,
                preset=row.preset,
                dataset=row.dataset,
                overrides=dict(row.overrides),
                snapshot_id=row.snapshotId,
                snapshot_name=row.snapshotName,
            )
            for row in request.runs
        ],
        snapshot_revisions=tuple(
            ConfigSnapshotRevision(
                id=revision.id,
                semantic_revision=revision.semanticRevision,
            )
            for revision in request.snapshotRevisions
        ),
    )


def _run_change_to_payload(change: TrainingRunChangeView) -> dict[str, Any]:
    return {
        "key": change.key,
        "label": change.label,
        "value": change.value,
        "source": change.source,
    }


def _run_to_payload(run: TrainingRunView) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": run.id,
        "index": run.index,
        "status": run.status,
        "preset": run.preset,
        "dataset": run.dataset,
        "experimentTask": run.experiment_task,
        "changes": [_run_change_to_payload(change) for change in run.changes],
        "overrides": run.overrides,
        "command": run.command,
        "commandArgv": run.command_argv,
        "commands": {
            "posix": run.commands.posix,
            "powershell": run.commands.powershell,
        },
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


def _summary_to_payload(summary: TrainingRunPlanSummaryView) -> dict[str, int]:
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


def run_plan_to_payload(plan: TrainingRunPlanView) -> dict[str, Any]:
    return {
        **_model_identity(plan.model),
        "preset": plan.preset,
        "presets": plan.presets,
        "experimentTask": plan.experiment_task,
        "datasets": plan.datasets,
        "overrides": plan.overrides,
        "search": search_to_payload(plan.search) if plan.search is not None else None,
        "logFolder": plan.log_folder,
        "isRandomSearch": plan.is_random_search,
        "runs": [_run_to_payload(run) for run in plan.runs],
        "summary": _summary_to_payload(plan.summary),
        "snapshotRevisions": [
            {
                "id": revision.id,
                "semanticRevision": revision.semantic_revision,
            }
            for revision in plan.snapshot_revisions
        ],
    }


__all__ = [
    "run_plan_to_payload",
    "search_from_request",
    "search_to_payload",
    "submitted_plan_from_request",
]
