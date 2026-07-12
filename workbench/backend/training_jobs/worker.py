"""Execute one persisted Training Job worker payload."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.model_packages import (
    ModelPackage,
    model_id_from_payload,
    model_identity_payload_from_id,
    model_package,
)
from emperor.runs import (
    FilesystemRunArtifacts,
    JsonlTrainingProgressCallback,
    execute_runs,
)

from workbench.backend.training_jobs.launcher import TRAINING_LOGS_ROOT_ENV
from workbench.backend.training_jobs.run_plan_adapter import (
    accept_worker_run_plan,
    worker_payload_names,
)

WORKBENCH_PROGRESS_STEP_INTERVAL = 25


def load_model_parts(model_id: str) -> ModelPackage:
    package = model_package(model_id)
    if package is None:
        raise ValueError(f"Unknown model: {model_id}")
    return package


def _payload_model_id(payload: Mapping[str, Any]) -> str:
    model_id = model_id_from_payload(payload)
    if model_id is None:
        raise ValueError("Training payload does not include a valid model identity.")
    return model_id


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
        raw_plan.get("experimentTask")
        if isinstance(raw_plan, Mapping)
        else None
    )
    try:
        datasets, monitors = worker_payload_names(payload)
        if isinstance(raw_plan, Mapping):
            model_id = _payload_model_id(raw_plan)
            identity = model_identity_payload_from_id(model_id)
            preset = raw_plan.get("preset")
            presets = raw_plan.get("presets") or [preset]
        else:
            model_id = ""
            identity = {}
            preset = None
            presets = []
        started_event = {
            "type": "started",
            "status": "running",
            "jobId": payload["id"],
            **identity,
            "preset": preset,
            "presets": presets,
            "datasets": datasets,
            "monitors": monitors,
        }
        if experiment_task is not None:
            started_event["experimentTask"] = experiment_task
        progress.write_event(started_event)

        if not isinstance(raw_plan, Mapping):
            raise ValueError(
                "Training payload does not include a non-empty materialized "
                "run plan."
            )
        package = load_model_parts(model_id)
        plan = accept_worker_run_plan(package, payload)
        execute_runs(
            package,
            plan,
            artifacts=FilesystemRunArtifacts(
                root=Path(os.environ.get(TRAINING_LOGS_ROOT_ENV, "logs")),
                namespace=raw_plan.get("logFolder") or None,
            ),
            progress=progress,
            monitors=tuple(monitors),
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
