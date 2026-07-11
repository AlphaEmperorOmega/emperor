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
    PlanningBudget,
    RunRequest,
    RunsError,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
    accept_run_plan,
    execute_runs,
)

from workbench.backend.training_jobs.launcher import TRAINING_LOGS_ROOT_ENV
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
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


def _bounded_payload_names(
    payload: Mapping[str, Any],
    field: str,
    *,
    limit: int,
    required: bool,
) -> list[str]:
    raw_names = payload.get(field)
    if raw_names is None:
        if required:
            raise ValueError(f"Training worker payload requires {field}.")
        return []
    if not isinstance(raw_names, list) or (required and not raw_names):
        requirement = "a non-empty list" if required else "a list"
        raise ValueError(f"Training worker payload {field} must be {requirement}.")
    if len(raw_names) > limit:
        raise ValueError(
            f"Training worker payload accepts at most {limit} {field}."
        )
    if any(not isinstance(name, str) or not name.strip() for name in raw_names):
        raise ValueError(
            f"Training worker payload {field} must contain non-empty strings."
        )
    return raw_names


def _bounded_payload_collections(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    return (
        _bounded_payload_names(
            payload,
            "datasets",
            limit=MAX_TRAINING_DATASETS,
            required=True,
        ),
        _bounded_payload_names(
            payload,
            "monitors",
            limit=MAX_TRAINING_MONITORS,
            required=False,
        ),
    )


def _search_spec(raw_search: object) -> SearchSpec | None:
    if raw_search is None:
        return None
    if not isinstance(raw_search, Mapping):
        raise ValueError("Training search must be an object.")
    raw_values = raw_search.get("values")
    if not isinstance(raw_values, Mapping) or not raw_values:
        raise ValueError("Training search requires at least one selected axis.")
    if len(raw_values) > MAX_TRAINING_SEARCH_AXES:
        raise ValueError(
            f"Training search accepts at most {MAX_TRAINING_SEARCH_AXES} "
            "selected axes."
        )
    mode = raw_search.get("mode")
    random_samples = raw_search.get("randomSamples")
    if mode == "random" and (
        isinstance(random_samples, bool)
        or not isinstance(random_samples, int)
        or random_samples < 1
        or random_samples > MAX_TRAINING_PLANNED_RUNS
    ):
        raise ValueError(
            "Random search sample count must be an integer between 1 and "
            f"{MAX_TRAINING_PLANNED_RUNS}."
        )
    axes: list[SearchAxisSelection] = []
    for key, values in raw_values.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search axis '{key}' requires at least one selected value."
            )
        if len(values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise ValueError(
                f"Search axis '{key}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )
        axes.append(
            SearchAxisSelection(
                key=str(key),
                values=tuple(values),
            )
        )
    return SearchSpec(
        mode=str(mode),  # type: ignore[arg-type]
        axes=tuple(axes),
        random_samples=random_samples,  # type: ignore[arg-type]
    )


def _run_request(payload: Mapping[str, Any]) -> RunRequest:
    raw_presets = payload.get("presets") or [payload.get("preset")]
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_presets, list):
        raise ValueError("Training payload presets must be a list.")
    if not isinstance(raw_datasets, list):
        raise ValueError("Training payload datasets must be a list.")
    raw_overrides = payload.get("overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("Training payload overrides must be an object.")
    experiment_task = payload.get("experimentTask")
    return RunRequest(
        presets=tuple(str(preset) for preset in raw_presets if preset is not None),
        datasets=tuple(str(dataset) for dataset in raw_datasets),
        experiment_task=(
            str(experiment_task) if experiment_task is not None else None
        ),
        overrides=dict(raw_overrides),
        search=_search_spec(payload.get("search")),
    )


def _require_run_plan_rows(payload: Mapping[str, Any]) -> list[Any]:
    run_plan = payload.get("runPlan")
    if not isinstance(run_plan, Mapping):
        raise ValueError(
            "Training payload does not include a non-empty materialized run plan."
        )
    rows = run_plan.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Training payload does not include a non-empty materialized run plan."
        )
    if len(rows) > MAX_TRAINING_PLANNED_RUNS:
        raise ValueError(
            "Submitted run plan is too large: "
            f"{len(rows)} submitted runs exceeds "
            f"{MAX_TRAINING_PLANNED_RUNS}."
        )
    return rows


def _submitted_runs(
    payload: Mapping[str, Any],
    rows: list[Any],
) -> tuple[SubmittedRun, ...]:
    payload_task = payload.get("experimentTask") or None
    submitted: list[SubmittedRun] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Run plan row {index} must be an object.")
        run_id = row.get("id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"Run plan row {index} requires a non-empty id.")
        preset = row.get("preset")
        if not isinstance(preset, str) or not preset.strip():
            raise ValueError(f"Run plan row {index} requires a non-empty preset.")
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"Run plan row {index} requires a non-empty dataset.")
        if "experimentTask" not in row:
            raise ValueError(f"Run plan row {index} requires experimentTask.")
        row_task = row.get("experimentTask")
        if row_task != payload_task:
            raise ValueError(
                "Submitted run plan contains an experimentTask that does not "
                "match the training job experimentTask."
            )
        if "overrides" not in row:
            raise ValueError(f"Run plan row {index} requires overrides.")
        overrides = row.get("overrides")
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Run plan row {index} overrides must be an object.")
        submitted.append(
            SubmittedRun(
                id=run_id,
                preset=preset,
                dataset=dataset,
                overrides=dict(overrides),
            )
        )
    return tuple(submitted)


def _validate_plan_envelope(
    model_id: str,
    payload: Mapping[str, Any],
) -> None:
    run_plan = payload.get("runPlan")
    if not isinstance(run_plan, Mapping):
        return
    envelope_fields = (
        ("preset", "primary preset"),
        ("presets", "presets"),
        ("experimentTask", "experiment task"),
        ("datasets", "datasets"),
        ("overrides", "overrides"),
        ("search", "search"),
        ("logFolder", "log folder"),
    )
    for field, label in envelope_fields:
        if field not in payload:
            raise ValueError(f"Training payload requires {label}.")
        if field not in run_plan:
            raise ValueError(f"Run plan requires {label}.")
    if not isinstance(payload["preset"], str) or not payload["preset"].strip():
        raise ValueError("Training payload primary preset must be non-empty.")
    if not isinstance(payload["presets"], list) or not payload["presets"]:
        raise ValueError("Training payload presets must be a non-empty list.")
    if not all(
        isinstance(preset, str) and preset.strip()
        for preset in payload["presets"]
    ):
        raise ValueError("Training payload presets must contain non-empty strings.")
    if (
        not isinstance(payload["experimentTask"], str)
        or not payload["experimentTask"].strip()
    ):
        raise ValueError("Training payload experiment task must be non-empty.")
    if not isinstance(payload["datasets"], list) or not payload["datasets"]:
        raise ValueError("Training payload datasets must be a non-empty list.")
    if not all(
        isinstance(dataset, str) and dataset.strip()
        for dataset in payload["datasets"]
    ):
        raise ValueError("Training payload datasets must contain non-empty strings.")
    if not isinstance(payload["overrides"], Mapping):
        raise ValueError("Training payload overrides must be an object.")
    if payload["search"] is not None and not isinstance(
        payload["search"], Mapping
    ):
        raise ValueError("Training payload search must be an object or null.")
    if not isinstance(payload["logFolder"], str):
        raise ValueError("Training payload log folder must be a string.")
    plan_model_id = model_id_from_payload(run_plan)
    if plan_model_id is None:
        raise ValueError("Run plan does not include a valid model identity.")
    if plan_model_id != model_id:
        raise ValueError(
            f"Run plan model '{plan_model_id}' does not match selected model "
            f"'{model_id}'."
        )
    plan_task = run_plan.get("experimentTask")
    payload_task = payload.get("experimentTask")
    if plan_task != payload_task:
        raise ValueError(
            f"Run plan experiment task '{plan_task}' does not match training "
            f"job task '{payload_task}'."
        )
    for field, label in envelope_fields:
        if run_plan.get(field) != payload.get(field):
            raise ValueError(
                f"Run plan {label} does not match the training job {label}."
            )


def _accepted_plan(
    package: ModelPackage,
    payload: Mapping[str, Any],
):
    _bounded_payload_collections(payload)
    rows = _require_run_plan_rows(payload)
    _validate_plan_envelope(package.catalog_key, payload)
    request = _run_request(payload)
    submitted_runs = _submitted_runs(payload, rows)
    try:
        return accept_run_plan(
            package,
            request,
            submitted_runs,
            budget=PlanningBudget(
                max_axes=MAX_TRAINING_SEARCH_AXES,
                max_values_per_axis=MAX_TRAINING_SEARCH_AXIS_VALUES,
                max_materialized_runs=MAX_TRAINING_PLANNED_RUNS,
            ),
        )
    except RunsError as exc:
        raise ValueError(str(exc)) from exc


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
    model_id = _payload_model_id(payload)
    identity = model_identity_payload_from_id(model_id)
    progress = JsonlTrainingProgressCallback(
        progress_path,
        step_interval=WORKBENCH_PROGRESS_STEP_INTERVAL,
    )
    experiment_task = payload.get("experimentTask") or None
    try:
        datasets, monitors = _bounded_payload_collections(payload)
        started_event = {
            "type": "started",
            "status": "running",
            "jobId": payload["id"],
            **identity,
            "preset": payload["preset"],
            "presets": payload.get("presets") or [payload["preset"]],
            "datasets": datasets,
            "monitors": monitors,
        }
        if experiment_task is not None:
            started_event["experimentTask"] = experiment_task
        progress.write_event(started_event)

        package = load_model_parts(model_id)
        plan = _accepted_plan(package, payload)
        execute_runs(
            package,
            plan,
            artifacts=FilesystemRunArtifacts(
                root=Path(os.environ.get(TRAINING_LOGS_ROOT_ENV, "logs")),
                namespace=payload.get("logFolder") or None,
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
