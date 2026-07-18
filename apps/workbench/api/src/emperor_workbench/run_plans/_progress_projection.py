from __future__ import annotations

from dataclasses import replace
from typing import Any

from emperor_workbench.model_packages import normalize_preset_token
from emperor_workbench.run_plans._records import (
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
    TrainingRunView,
)


def _summarize(runs: list[TrainingRunView]) -> TrainingRunPlanSummaryView:
    statuses = [run.status for run in runs]
    completed_epochs = 0
    remaining_epochs = 0
    total_epochs = 0
    for run in runs:
        run_total = run.total_epochs
        run_current = run.current_epoch
        run_status = run.status
        total_epochs += run_total
        if run_status == "Completed":
            run_done = run_total
        elif run_status in {"Running", "Failed", "Cancelled"}:
            run_done = min(run_current, run_total)
        else:
            run_done = 0
        completed_epochs += run_done
        if run_status in {"Pending", "Running"}:
            remaining_epochs += max(0, run_total - run_done)

    return TrainingRunPlanSummaryView(
        total_runs=len(runs),
        completed_runs=statuses.count("Completed"),
        running_runs=statuses.count("Running"),
        pending_runs=statuses.count("Pending"),
        failed_runs=statuses.count("Failed"),
        cancelled_runs=statuses.count("Cancelled"),
        skipped_runs=statuses.count("Skipped"),
        total_epochs=total_epochs,
        completed_epochs=completed_epochs,
        remaining_epochs=remaining_epochs,
    )


def _run_for_progress_event(
    *,
    event: dict[str, Any],
    runs: list[TrainingRunView],
    run_by_id: dict[str, int],
) -> int | None:
    run_id = event.get("runId")
    if isinstance(run_id, str) and run_id in run_by_id:
        return run_by_id[run_id]

    run_index = event.get("runIndex")
    if isinstance(run_index, int):
        if 1 <= run_index <= len(runs):
            return run_index - 1
        if 0 <= run_index < len(runs):
            return run_index

    dataset = event.get("dataset")
    preset = normalize_preset_token(event.get("preset"))
    if dataset is None:
        return None
    candidates = [
        index
        for index, run in enumerate(runs)
        if run.dataset == dataset
        and (
            preset is None
            or normalize_preset_token(run.preset) == normalize_preset_token(preset)
        )
    ]
    return next(
        (
            index
            for index in candidates
            if runs[index].status not in {"Completed", "Failed", "Cancelled"}
        ),
        candidates[0] if candidates else None,
    )


def _progress_event_epoch(event: dict[str, Any], total_epochs: int) -> int:
    raw_epoch = event.get("epoch")
    if not isinstance(raw_epoch, int):
        return 0
    return min(total_epochs, max(0, raw_epoch + 1))


class RunPlanProgressProjector:
    """Pure projections from Training Job progress events into a typed Run Plan."""

    @staticmethod
    def summarize(runs: list[TrainingRunView]) -> TrainingRunPlanSummaryView:
        return _summarize(runs)

    @staticmethod
    def index(runs: list[TrainingRunView]) -> dict[str, int]:
        return {run.id: index for index, run in enumerate(runs)}

    @staticmethod
    def apply(
        *,
        plan: TrainingRunPlanView,
        run_by_id: dict[str, int],
        event: dict[str, Any],
    ) -> TrainingRunPlanView:
        run_index = _run_for_progress_event(
            event=event,
            runs=plan.runs,
            run_by_id=run_by_id,
        )
        if run_index is None:
            return plan

        runs = list(plan.runs)
        run = runs[run_index]
        event_type = event.get("type")
        total_epochs = run.total_epochs
        changes: dict[str, Any] = {}
        if event.get("logDir"):
            changes["log_dir"] = str(event["logDir"])
        if isinstance(event.get("metrics"), dict):
            changes["metrics"] = dict(event["metrics"])

        if event_type == "dataset_started":
            changes["status"] = "Running"
            changes["current_epoch"] = max(0, run.current_epoch)
        elif event_type in {
            "epoch_started",
            "step",
            "validation",
            "fit_completed",
            "test_completed",
        }:
            changes["status"] = "Running"
            changes["current_epoch"] = max(
                run.current_epoch,
                _progress_event_epoch(event, total_epochs),
            )
        elif event_type == "dataset_completed":
            changes["status"] = "Completed"
            changes["current_epoch"] = total_epochs
        elif event_type == "error":
            changes["status"] = "Failed"
            changes["current_epoch"] = max(
                run.current_epoch,
                _progress_event_epoch(event, total_epochs),
            )
            changes["error"] = str(event.get("error") or "Training failed")
            if event.get("traceback"):
                changes["error_traceback"] = str(event.get("traceback"))

        if not changes:
            return plan
        runs[run_index] = replace(run, **changes)
        return replace(plan, runs=runs)

    @staticmethod
    def finalize(
        plan: TrainingRunPlanView,
        *,
        job_status: str,
        latest_failed_event: dict[str, Any] | None = None,
    ) -> TrainingRunPlanView:
        runs = list(plan.runs)
        latest_failed_event = latest_failed_event or {}
        if job_status == "cancelled":
            for index, run in enumerate(runs):
                if run.status == "Running":
                    runs[index] = replace(run, status="Cancelled")
                elif run.status == "Pending":
                    runs[index] = replace(run, status="Skipped")
        elif job_status == "failed":
            failed_seen = any(run.status == "Failed" for run in runs)
            for index, run in enumerate(runs):
                if run.status == "Running":
                    runs[index] = replace(run, status="Failed")
                    failed_seen = True
                elif run.status == "Pending":
                    if not failed_seen:
                        changes = {"status": "Failed", "error": "Training failed"}
                        if latest_failed_event.get("traceback"):
                            changes["error_traceback"] = str(
                                latest_failed_event.get("traceback")
                            )
                        runs[index] = replace(run, **changes)
                        failed_seen = True
                    else:
                        runs[index] = replace(run, status="Skipped")
        elif job_status == "completed":
            for index, run in enumerate(runs):
                if run.status == "Running":
                    runs[index] = replace(
                        run,
                        status="Completed",
                        current_epoch=run.total_epochs,
                    )

        return replace(plan, runs=runs, summary=_summarize(runs))


__all__ = ["RunPlanProgressProjector"]
