from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

SummaryCallback = Callable[[list[dict[str, Any]]], dict[str, int]]


def project_training_run_progress(
    run_plan: dict[str, Any],
    events: list[dict[str, Any]],
    job_status: str,
    summarize: SummaryCallback,
) -> dict[str, Any]:
    plan = copy.deepcopy(run_plan)
    runs = plan.get("runs") or []
    latest_failed_event = next(
        (event for event in reversed(events) if event.get("status") == "failed"),
        {},
    )
    run_by_id = {
        str(run.get("id")): run
        for run in runs
        if run.get("id") is not None
    }
    for event in events:
        row = _run_for_event(
            event=event,
            runs=runs,
            run_by_id=run_by_id,
        )
        if row is None:
            continue

        event_type = event.get("type")
        total_epochs = int(row.get("totalEpochs") or 0)
        if event.get("logDir"):
            row["logDir"] = event.get("logDir")
        if isinstance(event.get("metrics"), dict):
            row["metrics"] = event["metrics"]

        if event_type == "dataset_started":
            row["status"] = "Running"
            row["currentEpoch"] = max(0, int(row.get("currentEpoch") or 0))
        elif event_type in {
            "epoch_started",
            "step",
            "validation",
            "fit_completed",
            "test_completed",
        }:
            row["status"] = "Running"
            row["currentEpoch"] = max(
                int(row.get("currentEpoch") or 0),
                _event_epoch(event, total_epochs),
            )
        elif event_type == "dataset_completed":
            row["status"] = "Completed"
            row["currentEpoch"] = total_epochs
        elif event_type == "error":
            row["status"] = "Failed"
            row["currentEpoch"] = max(
                int(row.get("currentEpoch") or 0),
                _event_epoch(event, total_epochs),
            )
            row["error"] = str(event.get("error") or "Training failed")
            if event.get("traceback"):
                row["errorTraceback"] = str(event.get("traceback"))

    if job_status == "cancelled":
        for row in runs:
            if row.get("status") == "Running":
                row["status"] = "Cancelled"
            elif row.get("status") == "Pending":
                row["status"] = "Skipped"
    elif job_status == "failed":
        failed_seen = any(row.get("status") == "Failed" for row in runs)
        for row in runs:
            if row.get("status") == "Running":
                row["status"] = "Failed"
                failed_seen = True
            elif row.get("status") == "Pending":
                if not failed_seen:
                    row["status"] = "Failed"
                    row["error"] = "Training failed"
                    if latest_failed_event.get("traceback"):
                        row["errorTraceback"] = str(
                            latest_failed_event.get("traceback")
                        )
                    failed_seen = True
                else:
                    row["status"] = "Skipped"
    elif job_status == "completed":
        for row in runs:
            if row.get("status") == "Running":
                row["status"] = "Completed"
                row["currentEpoch"] = int(row.get("totalEpochs") or 0)

    plan["summary"] = summarize(runs)
    return plan


def _normalize_preset_token(preset: str | None) -> str | None:
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


def _event_preset_name(event: dict[str, Any]) -> str | None:
    return _normalize_preset_token(event.get("preset") or event.get("option"))


def _run_for_event(
    *,
    event: dict[str, Any],
    runs: list[dict[str, Any]],
    run_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    run_id = event.get("runId")
    if isinstance(run_id, str) and run_id in run_by_id:
        return run_by_id[run_id]

    run_index = event.get("runIndex")
    if isinstance(run_index, int):
        if 1 <= run_index <= len(runs):
            return runs[run_index - 1]
        if 0 <= run_index < len(runs):
            return runs[run_index]

    dataset = event.get("dataset")
    preset = _event_preset_name(event)
    if dataset is None:
        return None
    candidates = [
        run
        for run in runs
        if run.get("dataset") == dataset
        and (
            preset is None
            or _normalize_preset_token(str(run.get("preset")))
            == _normalize_preset_token(preset)
        )
    ]
    return next(
        (
            run
            for run in candidates
            if run.get("status") not in {"Completed", "Failed", "Cancelled"}
        ),
        candidates[0] if candidates else None,
    )


def _event_epoch(event: dict[str, Any], total_epochs: int) -> int:
    raw_epoch = event.get("epoch")
    if not isinstance(raw_epoch, int):
        return 0
    return min(total_epochs, max(0, raw_epoch + 1))
