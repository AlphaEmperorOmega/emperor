from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

from model_runtime.runs import JsonlTrainingProgressCallback, RunPlan

from emperor_workbench.run_plans import RunPlanWorkerAcceptance
from emperor_workbench.training_jobs._containment._launcher import (
    TRAINING_LOGS_ROOT_ENV,
)

WORKBENCH_PROGRESS_STEP_INTERVAL = 25


def execute_project_run_plan(
    payload: dict[str, object],
    *,
    logs_root: Path,
    progress_path: Path,
) -> RunPlan:
    return RunPlanWorkerAcceptance.execute(
        payload,
        logs_root=logs_root,
        progress_path=progress_path,
        progress_step_interval=WORKBENCH_PROGRESS_STEP_INTERVAL,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a workbench training job.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--progress", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload_path = Path(args.payload)
    progress_path = Path(args.progress)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training worker payload must be a JSON object.")
    progress = JsonlTrainingProgressCallback(
        progress_path,
        step_interval=WORKBENCH_PROGRESS_STEP_INTERVAL,
    )
    raw_plan = payload.get("runPlan")
    experiment_task = (
        raw_plan.get("experimentTask") if isinstance(raw_plan, dict) else None
    )
    try:
        context = RunPlanWorkerAcceptance.describe(payload)
        started_event = {
            "type": "started",
            "status": "running",
            "jobId": payload["id"],
            "modelType": context.model_type,
            "model": context.model_name,
            "preset": context.preset,
            "presets": list(context.presets),
            "datasets": list(context.datasets),
            "monitors": list(context.monitors),
        }
        if context.experiment_task:
            started_event["experimentTask"] = context.experiment_task
        progress.write_event(started_event)

        plan = execute_project_run_plan(
            payload,
            logs_root=Path(os.environ.get(TRAINING_LOGS_ROOT_ENV, "logs")),
            progress_path=progress_path,
        )
        completed_event = {
            "type": "completed",
            "status": "completed",
            "jobId": payload["id"],
            "preset": plan.presets[-1],
            "presets": list(plan.presets),
        }
        if experiment_task is not None:
            completed_event["experimentTask"] = experiment_task
        progress.write_event(completed_event)
    except Exception as exc:
        error_event = {
            "type": "error",
            "status": "failed",
            "jobId": payload.get("id"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if experiment_task is not None:
            error_event["experimentTask"] = experiment_task
        progress.write_event(error_event)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


__all__ = ["TRAINING_LOGS_ROOT_ENV", "main"]
