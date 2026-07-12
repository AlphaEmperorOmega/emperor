"""Canonical Workbench adaptation for authoritative Emperor Run Plans."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from emperor.inspection import (
    ConfigurationField,
    SearchAxis,
    resolve_override_key,
)
from emperor.model_packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_model_param,
    dataset_name,
    iter_supported_config_keys,
    model_id_from_payload,
    model_identity_payload_from_id,
    normalize_key,
    parse_config_value,
    serialize_config_value,
)
from emperor.runs import (
    PlanningBudget,
    RunPlan,
    RunRequest,
    RunsError,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
    accept_run_plan,
    plan_runs,
)

from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import is_valid_log_experiment_name
from workbench.backend.model_identity import normalize_preset_token
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)

if TYPE_CHECKING:
    from workbench.backend.training_jobs.contracts import (
        CreateTrainingJobCommand,
        CreateTrainingRunPlanCommand,
    )

ConfigValue = bool | int | float | str | None
TrainingRunStatus = Literal[
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
]
TrainingRunChangeSource = Literal["override", "search"]


@dataclass(frozen=True, slots=True)
class TrainingSearch:
    mode: Literal["grid", "random"]
    values: dict[str, list[ConfigValue]] = field(default_factory=dict)
    random_samples: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingRunChangeView:
    key: str
    label: str
    value: ConfigValue
    source: TrainingRunChangeSource


@dataclass(frozen=True, slots=True)
class TrainingRunView:
    id: str
    index: int
    status: TrainingRunStatus
    preset: str
    dataset: str
    experiment_task: str
    changes: list[TrainingRunChangeView]
    overrides: dict[str, Any]
    command: str
    total_epochs: int
    snapshot_id: str | None = None
    snapshot_name: str | None = None
    snapshot_id_present: bool = False
    snapshot_name_present: bool = False
    current_epoch: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    log_dir: str | None = None
    error: str | None = None
    error_traceback: str | None = None


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


@dataclass(frozen=True, slots=True)
class TrainingRunPlanView:
    model: str
    preset: str
    presets: list[str]
    experiment_task: str
    datasets: list[str]
    overrides: dict[str, Any]
    search: TrainingSearch | None
    log_folder: str
    is_random_search: bool
    runs: list[TrainingRunView]
    summary: TrainingRunPlanSummaryView


@dataclass(frozen=True, slots=True)
class SubmittedTrainingRun:
    """Authoritative row choices accepted from an untrusted caller."""

    id: str
    preset: str
    dataset: str
    overrides: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str | None = None
    snapshot_name: str | None = None


@dataclass(frozen=True, slots=True)
class SubmittedTrainingRunPlan:
    """Minimal submitted Run Plan; all presentation state is derived."""

    runs: list[SubmittedTrainingRun] = field(default_factory=list)


TrainingRunPlanDocument = dict[str, Any]


def decode_persisted_run_plan(payload: object) -> TrainingRunPlanDocument:
    """Decode the canonical mutable document used by job persistence/projection."""

    if not isinstance(payload, Mapping):
        raise ValueError("Persisted Run Plan must be an object.")
    rows = payload.get("runs")
    summary = payload.get("summary")
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ValueError("Persisted Run Plan rows must be a list of objects.")
    if not isinstance(summary, Mapping):
        raise ValueError("Persisted Run Plan summary must be an object.")
    return copy.deepcopy(dict(payload))


def encode_persisted_run_plan(
    document: Mapping[str, Any],
) -> TrainingRunPlanDocument:
    """Return an isolated JSON-ready persistence/worker representation."""

    return decode_persisted_run_plan(document)


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
            training_run_change_to_payload(change) for change in run.changes
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


def submitted_run_plan_from_payload(
    payload: Mapping[str, Any],
) -> SubmittedTrainingRunPlan:
    return SubmittedTrainingRunPlan(
        runs=[
            SubmittedTrainingRun(
                id=str(row.get("id") or ""),
                preset=str(row.get("preset") or ""),
                dataset=str(row.get("dataset") or ""),
                overrides=dict(row.get("overrides") or {}),
                snapshot_id=(
                    str(row["snapshotId"])
                    if row.get("snapshotId") is not None
                    else None
                ),
                snapshot_name=(
                    str(row["snapshotName"])
                    if row.get("snapshotName") is not None
                    else None
                ),
            )
            for row in _mapping_items(payload.get("runs"))
        ]
    )


def submitted_run_plan_to_payload(
    plan: SubmittedTrainingRunPlan,
) -> dict[str, Any]:
    return {
        "runs": [
            {
                "id": run.id,
                "preset": run.preset,
                "snapshotId": run.snapshot_id,
                "snapshotName": run.snapshot_name,
                "dataset": run.dataset,
                "overrides": run.overrides,
            }
            for run in plan.runs
        ]
    }


def _bounded_worker_names(
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
    return list(raw_names)


def worker_payload_names(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Defensively decode bounded names before the worker emits `started`."""

    raw_plan = payload.get("runPlan")
    datasets = (
        _bounded_worker_names(
            raw_plan,
            "datasets",
            limit=MAX_TRAINING_DATASETS,
            required=True,
        )
        if isinstance(raw_plan, Mapping)
        else []
    )
    return (
        datasets,
        _bounded_worker_names(
            payload,
            "monitors",
            limit=MAX_TRAINING_MONITORS,
            required=False,
        ),
    )


def _worker_search_spec(raw_search: object) -> SearchSpec | None:
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
        axes.append(SearchAxisSelection(key=str(key), values=tuple(values)))
    return SearchSpec(
        mode=str(mode),  # type: ignore[arg-type]
        axes=tuple(axes),
        random_samples=random_samples,  # type: ignore[arg-type]
    )


def _worker_run_request(run_plan: Mapping[str, Any]) -> RunRequest:
    raw_presets = run_plan.get("presets") or [run_plan.get("preset")]
    raw_datasets = run_plan.get("datasets")
    if not isinstance(raw_presets, list):
        raise ValueError("Training payload presets must be a list.")
    if not isinstance(raw_datasets, list):
        raise ValueError("Training payload datasets must be a list.")
    raw_overrides = run_plan.get("overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("Training payload overrides must be an object.")
    experiment_task = run_plan.get("experimentTask")
    return RunRequest(
        presets=tuple(str(preset) for preset in raw_presets if preset is not None),
        datasets=tuple(str(dataset) for dataset in raw_datasets),
        experiment_task=(
            str(experiment_task) if experiment_task is not None else None
        ),
        overrides=dict(raw_overrides),
        search=_worker_search_spec(run_plan.get("search")),
    )


def _worker_plan_rows(payload: Mapping[str, Any]) -> list[Any]:
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
            f"{len(rows)} submitted runs exceeds {MAX_TRAINING_PLANNED_RUNS}."
        )
    return rows


def _worker_submitted_runs(
    run_plan: Mapping[str, Any],
    rows: list[Any],
) -> tuple[SubmittedRun, ...]:
    plan_task = run_plan.get("experimentTask") or None
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
        if row.get("experimentTask") != plan_task:
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


def _validate_worker_plan_envelope(
    model_id: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    run_plan = payload.get("runPlan")
    if not isinstance(run_plan, Mapping):
        raise ValueError("Training payload runPlan must be an object.")
    required_fields = (
        ("preset", "primary preset"),
        ("presets", "presets"),
        ("experimentTask", "experiment task"),
        ("datasets", "datasets"),
        ("overrides", "overrides"),
        ("search", "search"),
        ("logFolder", "log folder"),
    )
    for field_name, label in required_fields:
        if field_name not in run_plan:
            raise ValueError(f"Run plan requires {label}.")
    if not isinstance(run_plan["preset"], str) or not run_plan["preset"].strip():
        raise ValueError("Run plan primary preset must be non-empty.")
    if not isinstance(run_plan["presets"], list) or not run_plan["presets"]:
        raise ValueError("Run plan presets must be a non-empty list.")
    if not all(
        isinstance(preset, str) and preset.strip()
        for preset in run_plan["presets"]
    ):
        raise ValueError("Run plan presets must contain non-empty strings.")
    if run_plan["preset"] != run_plan["presets"][0]:
        raise ValueError(
            "Run plan primary preset and presets must agree on the first value."
        )
    if (
        not isinstance(run_plan["experimentTask"], str)
        or not run_plan["experimentTask"].strip()
    ):
        raise ValueError("Run plan experiment task must be non-empty.")
    if not isinstance(run_plan["datasets"], list) or not run_plan["datasets"]:
        raise ValueError("Run plan datasets must be a non-empty list.")
    if not all(
        isinstance(dataset, str) and dataset.strip()
        for dataset in run_plan["datasets"]
    ):
        raise ValueError("Run plan datasets must contain non-empty strings.")
    if not isinstance(run_plan["overrides"], Mapping):
        raise ValueError("Run plan overrides must be an object.")
    if run_plan["search"] is not None and not isinstance(
        run_plan["search"], Mapping
    ):
        raise ValueError("Run plan search must be an object or null.")
    if not isinstance(run_plan["logFolder"], str):
        raise ValueError("Run plan log folder must be a string.")
    plan_model_id = model_id_from_payload(run_plan)
    if plan_model_id is None:
        raise ValueError("Run plan does not include a valid model identity.")
    if plan_model_id != model_id:
        raise ValueError(
            f"Run plan model '{plan_model_id}' does not match selected model "
            f"'{model_id}'."
        )
    return run_plan


def _validate_worker_authoritative_metadata(
    run_plan: Mapping[str, Any],
    rows: list[Any],
) -> None:
    presets = set(run_plan["presets"])
    datasets = set(run_plan["datasets"])
    envelope_overrides = dict(run_plan["overrides"])
    raw_search = run_plan["search"]
    search_values = (
        dict(raw_search.get("values") or {})
        if isinstance(raw_search, Mapping)
        else {}
    )
    expected_random = bool(
        isinstance(raw_search, Mapping) and raw_search.get("mode") == "random"
    )
    if run_plan.get("isRandomSearch") is not expected_random:
        raise ValueError(
            "Run plan random-search marker does not match its Search Metadata."
        )

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        if row.get("preset") not in presets:
            raise ValueError(
                f"Run plan presets do not contain row {index} preset."
            )
        if row.get("dataset") not in datasets:
            raise ValueError(
                f"Run plan datasets do not contain row {index} dataset."
            )
        row_overrides = row.get("overrides")
        if not isinstance(row_overrides, Mapping):
            continue
        normalized_row_overrides = {
            normalize_key(str(key)): value
            for key, value in row_overrides.items()
        }
        for key, value in envelope_overrides.items():
            if normalized_row_overrides.get(normalize_key(str(key))) != value:
                raise ValueError(
                    f"Run plan overrides do not match row {index} overrides."
                )
        for key, values in search_values.items():
            row_value = normalized_row_overrides.get(normalize_key(str(key)))
            if not isinstance(values, list) or row_value not in values:
                raise ValueError(
                    f"Run plan search does not match row {index} overrides."
                )


def summarize_training_runs(runs: list[dict[str, Any]]) -> dict[str, int]:
    """Derive the complete Run Plan summary from exact projected rows."""

    statuses = [str(run.get("status", "Pending")) for run in runs]
    completed_epochs = 0
    remaining_epochs = 0
    total_epochs = 0
    for run in runs:
        row_total = int(run.get("totalEpochs") or 0)
        row_current = int(run.get("currentEpoch") or 0)
        row_status = str(run.get("status", "Pending"))
        total_epochs += row_total
        if row_status == "Completed":
            row_done = row_total
        elif row_status in {"Running", "Failed", "Cancelled"}:
            row_done = min(row_current, row_total)
        else:
            row_done = 0
        completed_epochs += row_done
        if row_status in {"Pending", "Running"}:
            remaining_epochs += max(0, row_total - row_done)

    return {
        "totalRuns": len(runs),
        "completedRuns": statuses.count("Completed"),
        "runningRuns": statuses.count("Running"),
        "pendingRuns": statuses.count("Pending"),
        "failedRuns": statuses.count("Failed"),
        "cancelledRuns": statuses.count("Cancelled"),
        "skippedRuns": statuses.count("Skipped"),
        "totalEpochs": total_epochs,
        "completedEpochs": completed_epochs,
        "remainingEpochs": remaining_epochs,
    }


def run_lookup_by_id(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(run.get("id")): run
        for run in runs
        if run.get("id") is not None
    }


def apply_training_run_progress_event(
    *,
    runs: list[dict[str, Any]],
    run_by_id: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    row = _run_for_progress_event(
        event=event,
        runs=runs,
        run_by_id=run_by_id,
    )
    if row is None:
        return

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
            _progress_event_epoch(event, total_epochs),
        )
    elif event_type == "dataset_completed":
        row["status"] = "Completed"
        row["currentEpoch"] = total_epochs
    elif event_type == "error":
        row["status"] = "Failed"
        row["currentEpoch"] = max(
            int(row.get("currentEpoch") or 0),
            _progress_event_epoch(event, total_epochs),
        )
        row["error"] = str(event.get("error") or "Training failed")
        if event.get("traceback"):
            row["errorTraceback"] = str(event.get("traceback"))


def finalize_training_run_progress(
    plan: dict[str, Any],
    *,
    job_status: str,
    latest_failed_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    runs = plan.get("runs") or []
    latest_failed_event = latest_failed_event or {}
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

    plan["summary"] = summarize_training_runs(runs)
    return plan


def _run_for_progress_event(
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
    preset = normalize_preset_token(event.get("preset"))
    if dataset is None:
        return None
    candidates = [
        run
        for run in runs
        if run.get("dataset") == dataset
        and (
            preset is None
            or normalize_preset_token(str(run.get("preset")))
            == normalize_preset_token(preset)
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


def _progress_event_epoch(event: dict[str, Any], total_epochs: int) -> int:
    raw_epoch = event.get("epoch")
    if not isinstance(raw_epoch, int):
        return 0
    return min(total_epochs, max(0, raw_epoch + 1))


def _require_package(model: str) -> ModelPackage:
    return WorkbenchInspectionAdapter.select(model).package


def _parse_workbench_search_value(
    package: ModelPackage,
    axis: SearchAxis,
    raw_value: Any,
) -> Any:
    if raw_value is None:
        return None
    try:
        parsed_value = parse_config_value(
            package.metadata.search_space_module,
            axis.search_key,
            str(raw_value),
        )
        if isinstance(parsed_value, type):
            abstract_error = abstract_config_class_error(parsed_value)
            if abstract_error is not None:
                raise ValueError(abstract_error)
        return serialize_config_value(parsed_value)
    except Exception as exc:
        raise InspectorError(
            f"Invalid search value for axis '{axis.key}': {raw_value!r}. {exc}"
        ) from exc


def _deduplicate_workbench_search_values(values: list[Any]) -> tuple[Any, ...]:
    deduplicated: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return tuple(deduplicated)


def _adapt_workbench_search(
    package: ModelPackage,
    preset_name: str,
    search: dict[str, Any] | None,
) -> tuple[SearchSpec | None, set[str]]:
    """Normalize the legacy HTTP shape without owning Run materialization."""

    if not search:
        return None, set()
    mode = search.get("mode")
    if mode not in {"grid", "random"}:
        raise InspectorError("Training search mode must be 'grid' or 'random'.")
    values_payload = search.get("values")
    if not isinstance(values_payload, dict) or not values_payload:
        raise InspectorError("Training search requires at least one selected axis.")
    if len(values_payload) > MAX_TRAINING_SEARCH_AXES:
        raise InspectorError(
            f"Training search accepts at most {MAX_TRAINING_SEARCH_AXES} "
            "selected axes."
        )

    random_samples: int | None = None
    if mode == "random":
        raw_samples = search.get("randomSamples", 10)
        if isinstance(raw_samples, bool) or not isinstance(raw_samples, int):
            raise InspectorError("Random search sample count must be an integer.")
        if raw_samples < 1:
            raise InspectorError("Random search sample count must be at least 1.")
        random_samples = raw_samples

    semantic_search = WorkbenchInspectionAdapter.from_package(
        package
    ).search_space(preset_name)
    axes_by_key = {normalize_key(axis.key): axis for axis in semantic_search.axes}
    ordered_keys: list[str] = []
    selections: dict[str, SearchAxisSelection] = {}
    model_params: set[str] = set()
    for raw_key, raw_values in values_payload.items():
        axis = axes_by_key.get(normalize_key(str(raw_key)))
        if axis is None:
            raise InspectorError(f"Unknown search axis '{raw_key}'.")
        if axis.locked:
            raise InspectorError(
                f"Search axis '{axis.key}' is locked by preset '{preset_name}'."
            )
        if not isinstance(raw_values, list) or not raw_values:
            raise InspectorError(
                f"Search axis '{axis.key}' requires at least one selected value."
            )
        if len(raw_values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise InspectorError(
                f"Search axis '{axis.key}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )

        serialized_values = _deduplicate_workbench_search_values(
            [
                _parse_workbench_search_value(package, axis, raw_value)
                for raw_value in raw_values
            ]
        )
        allowed_values = {serialize_config_value(value) for value in axis.values}
        invalid_values = [
            value for value in serialized_values if value not in allowed_values
        ]
        if invalid_values:
            raise InspectorError(
                f"Search axis '{axis.key}' received values outside its "
                f"search space: {invalid_values}."
            )

        canonical_key = normalize_key(axis.key)
        if canonical_key not in selections:
            ordered_keys.append(canonical_key)
        selections[canonical_key] = SearchAxisSelection(
            key=axis.key,
            values=serialized_values,
        )
        model_params.add(config_key_to_model_param(axis.key))

    return (
        SearchSpec(
            mode=mode,
            axes=tuple(selections[key] for key in ordered_keys),
            random_samples=random_samples,
        ),
        model_params,
    )


def _workbench_search_payload(search: SearchSpec | None) -> dict[str, Any] | None:
    if search is None:
        return None
    payload: dict[str, Any] = {
        "mode": search.mode,
        "values": {
            axis.key: list(axis.values or ())
            for axis in (search.axes or ())
        },
    }
    if search.random_samples is not None:
        payload["randomSamples"] = search.random_samples
    return payload


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_@%+=:,./-"
    )
    if all(character in safe for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _command_value(value: Any) -> str:
    serialized = serialize_config_value(value)
    if serialized is None:
        return "None"
    if isinstance(serialized, bool):
        return str(serialized).lower()
    return str(serialized)


def _field_label(field: ConfigurationField) -> str:
    return field.key.lower().replace("_", " ")


def _build_training_command(
    *,
    fields: tuple[ConfigurationField, ...],
    by_key: dict[str, ConfigurationField],
    model: str,
    preset: str,
    experiment_task: str,
    dataset: str,
    overrides: dict[str, Any],
    log_folder: str,
    monitors: list[str],
) -> str:
    values_by_field_key: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        field = by_key.get(normalize_key(str(raw_key)))
        if field is not None:
            values_by_field_key[field.key] = raw_value

    identity = model_identity_payload_from_id(model)
    parts = [
        "source",
        "experiment.sh",
        "--model-type",
        _shell_quote(identity["modelType"]),
        "--model",
        _shell_quote(identity["model"]),
        "--preset",
        _shell_quote(preset),
        "--experiment-task",
        _shell_quote(experiment_task),
        "--datasets",
        _shell_quote(dataset),
    ]
    if log_folder:
        parts.extend(["--logdir", _shell_quote(log_folder)])
    if monitors:
        parts.append("--monitors")
        parts.extend(_shell_quote(monitor) for monitor in monitors)

    config_parts: list[str] = []
    for field in fields:
        field_key = field.key
        if field_key not in values_by_field_key:
            continue
        config_parts.extend(
            [
                field.flag,
                _shell_quote(_command_value(values_by_field_key[field_key])),
            ]
        )
    if config_parts:
        parts.append("--config")
        parts.extend(config_parts)
    return " ".join(parts)


@dataclass(frozen=True)
class SelectedTrainingInputs:
    parts: ModelPackage
    request: RunRequest


@dataclass(frozen=True, slots=True)
class MaterializedTrainingRunPlan:
    document: TrainingRunPlanDocument
    monitors: tuple[str, ...]


class WorkbenchRunPlanAdapter:
    def __init__(self, random_source: random.Random | None = None) -> None:
        self._random = random_source or random

    def resolve_inputs(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        experiment_task: str | None = None,
        datasets: list[str],
        overrides: dict[str, Any],
        search: dict[str, Any] | None,
    ) -> SelectedTrainingInputs:
        if not datasets:
            raise InspectorError("Training requires at least one selected dataset.")

        parts = _require_package(model)
        try:
            selected_experiment_task = parts.resolve_experiment_task(experiment_task)
            selected_datasets = parts.resolve_datasets(
                datasets,
                selected_experiment_task,
            )
        except ValueError as exc:
            raise InspectorError(str(exc)) from exc
        selected_experiment_task_name = parts.task_name(selected_experiment_task)
        selected_preset_names = self._resolve_presets(
            parts,
            model,
            preset,
            presets,
        )
        parsed_search, search_model_params = _adapt_workbench_search(
            parts,
            selected_preset_names[0],
            search=search,
        )
        effective_overrides = self._effective_overrides_for_search(
            parts=parts,
            overrides=overrides,
            search_model_params=search_model_params,
        )
        self._parse_and_validate_overrides(
            parts=parts,
            selected_preset_names=selected_preset_names,
            effective_overrides=effective_overrides,
        )
        return SelectedTrainingInputs(
            parts=parts,
            request=RunRequest(
                presets=tuple(selected_preset_names),
                datasets=tuple(dataset_name(dataset) for dataset in selected_datasets),
                experiment_task=selected_experiment_task_name,
                overrides=effective_overrides,
                search=parsed_search,
            ),
        )

    def _effective_overrides_for_search(
        self,
        *,
        parts,
        overrides: dict[str, Any],
        search_model_params: set[str],
    ) -> dict[str, Any]:
        if not overrides or not search_model_params:
            return dict(overrides)
        supported = {
            normalize_key(config_key): config_key
            for config_key in iter_supported_config_keys(parts.runtime_defaults)
        }
        filtered: dict[str, Any] = {}
        for raw_key, raw_value in overrides.items():
            canonical_key, _legacy_residual_flag = resolve_override_key(
                normalize_key(str(raw_key)),
                supported,
            )
            if (
                canonical_key is not None
                and config_key_to_model_param(canonical_key)
                in search_model_params
            ):
                continue
            filtered[raw_key] = raw_value
        return filtered

    def _parse_and_validate_overrides(
        self,
        *,
        parts,
        selected_preset_names: list[str],
        effective_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = WorkbenchInspectionAdapter.from_package(parts)
        parsed_overrides = adapter.parse_overrides(
            effective_overrides,
        ).values
        for selected_preset in selected_preset_names:
            adapter.reject_locked_overrides(selected_preset, parsed_overrides)
        return dict(parsed_overrides)

    def valid_plan_log_folder(self, log_folder: str) -> str:
        return (
            log_folder
            if log_folder and is_valid_log_experiment_name(log_folder)
            else ""
        )

    def resolve_monitor_names(
        self,
        package: ModelPackage,
        monitor_names: list[str] | None,
    ) -> list[str]:
        try:
            return [
                monitor.name
                for monitor in package.resolve_monitors(monitor_names)
            ]
        except ValueError as exc:
            raise InspectorError(str(exc)) from exc

    def create_for_request(
        self,
        *,
        model: str,
        preset: str,
        presets: list[str] | None,
        experiment_task: str | None,
        datasets: list[str],
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str] | None,
        search: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selected = self.resolve_inputs(
            model=model,
            preset=preset,
            presets=presets,
            experiment_task=experiment_task,
            datasets=datasets,
            overrides=overrides,
            search=search,
        )
        monitor_names = self.resolve_monitor_names(selected.parts, monitors)
        return self.create(
            model=model,
            selected=selected,
            log_folder=self.valid_plan_log_folder(log_folder),
            monitors=monitor_names,
        )

    def summarize(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        return summarize_training_runs(runs)

    def create(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        log_folder: str,
        monitors: list[str] | None = None,
    ) -> dict[str, Any]:
        monitor_names = monitors or []
        try:
            semantic_plan = plan_runs(
                selected.parts,
                selected.request,
                random_source=(
                    self._random
                    if selected.request.search is not None
                    and selected.request.search.mode == "random"
                    else None
                ),
                budget=PlanningBudget(
                    max_axes=MAX_TRAINING_SEARCH_AXES,
                    max_values_per_axis=MAX_TRAINING_SEARCH_AXIS_VALUES,
                    max_materialized_runs=MAX_TRAINING_PLANNED_RUNS,
                ),
            )
        except RunsError as exc:
            raise InspectorError(str(exc)) from exc
        runs = [
            self._pending_semantic_run(
                model=model,
                parts=selected.parts,
                run=run,
                index=index,
                log_folder=log_folder,
                monitors=monitor_names,
                search=semantic_plan.search,
            )
            for index, run in enumerate(semantic_plan.runs, start=1)
        ]
        return self._plan_payload(
            model=model,
            semantic_plan=semantic_plan,
            log_folder=log_folder,
            runs=runs,
        )

    def _pending_semantic_run(
        self,
        *,
        model: str,
        parts: ModelPackage,
        run: RunSpec,
        index: int,
        log_folder: str,
        monitors: list[str],
        search: SearchSpec | None = None,
    ) -> dict[str, Any]:
        _fields, by_key = self._field_maps(model, run.preset)
        search_keys = {
            normalize_key(axis.key)
            for axis in (search.axes or ())
        } if search is not None else set()
        parameters = list(run.parameters)
        if search is not None:
            search_positions = {
                normalize_key(axis.key): index
                for index, axis in enumerate(search.axes or ())
            }
            fixed_parameters = [
                parameter
                for parameter in parameters
                if normalize_key(parameter.key) not in search_keys
            ]
            searched_parameters = sorted(
                (
                    parameter
                    for parameter in parameters
                    if normalize_key(parameter.key) in search_keys
                ),
                key=lambda parameter: search_positions[
                    normalize_key(parameter.key)
                ],
            )
            parameters = fixed_parameters + searched_parameters
        changes = []
        overrides: dict[str, Any] = {}
        for parameter in parameters:
            field = by_key.get(normalize_key(parameter.key))
            field_key = field.key if field is not None else parameter.key
            overrides[field_key] = parameter.value
            changes.append(
                {
                    "key": field_key,
                    "label": _field_label(field) if field is not None else field_key,
                    "value": parameter.value,
                    "source": (
                        "search"
                        if parameter.source == "search"
                        or normalize_key(parameter.key) in search_keys
                        else "override"
                    ),
                }
            )
        payload = self._pending_run(
            model=model,
            index=index,
            preset=run.preset,
            experiment_task=run.experiment_task,
            dataset=run.dataset,
            changes=changes,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
            total_epochs=self._run_total_epochs(parts, run),
        )
        payload["id"] = run.id
        return payload

    def from_submitted(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        run_plan: SubmittedTrainingRunPlan | Mapping[str, Any],
        log_folder: str,
        monitors: list[str] | None = None,
    ) -> dict[str, Any]:
        monitor_names = monitors or []
        submitted_plan = (
            run_plan
            if isinstance(run_plan, SubmittedTrainingRunPlan)
            else submitted_run_plan_from_payload(run_plan)
        )
        submitted_runs = submitted_run_plan_to_payload(submitted_plan)["runs"]
        if len(submitted_runs) > MAX_TRAINING_PLANNED_RUNS:
            raise InspectorError(
                "Submitted run plan is too large: "
                f"{len(submitted_runs)} submitted runs exceeds "
                f"{MAX_TRAINING_PLANNED_RUNS}."
            )
        try:
            semantic_plan = accept_run_plan(
                selected.parts,
                selected.request,
                tuple(
                    SubmittedRun(
                        id=(
                            str(row.get("id"))
                            if row.get("id")
                            else None
                        ),
                        preset=str(row.get("preset") or ""),
                        dataset=str(row.get("dataset") or ""),
                        overrides=dict(row.get("overrides") or {}),
                    )
                    for row in submitted_runs
                ),
                budget=PlanningBudget(
                    max_axes=MAX_TRAINING_SEARCH_AXES,
                    max_values_per_axis=MAX_TRAINING_SEARCH_AXIS_VALUES,
                    max_materialized_runs=MAX_TRAINING_PLANNED_RUNS,
                ),
            )
        except RunsError as exc:
            raise InspectorError(str(exc)) from exc

        runs = []
        for index, (row, semantic_run) in enumerate(
            zip(submitted_runs, semantic_plan.runs, strict=True),
            start=1,
        ):
            snapshot_id = row.get("snapshotId")
            snapshot_name = row.get("snapshotName")
            projected_row = self._pending_semantic_run(
                model=model,
                parts=selected.parts,
                run=semantic_run,
                index=index,
                log_folder=log_folder,
                monitors=monitor_names,
                search=semantic_plan.search,
            )
            runs.append(
                {
                    **projected_row,
                    "snapshotId": str(snapshot_id) if snapshot_id is not None else None,
                    "snapshotName": str(snapshot_name)
                    if snapshot_name is not None
                    else None,
                }
            )

        return self._plan_payload(
            model=model,
            semantic_plan=semantic_plan,
            log_folder=log_folder,
            runs=runs,
        )

    def _resolve_presets(
        self,
        parts,
        model: str,
        preset: str,
        presets: list[str] | None,
    ):
        raw_presets = presets if presets else [preset]
        selected = []
        seen = set()
        unknown = []
        for raw_preset in raw_presets:
            if not isinstance(raw_preset, str) or not raw_preset.strip():
                continue
            try:
                preset_member = parts.resolve_preset(raw_preset)
            except ValueError:
                unknown.append(raw_preset)
                continue
            if preset_member.name in seen:
                continue
            seen.add(preset_member.name)
            selected.append(
                (
                    parts.preset_name(preset_member),
                    preset_member,
                )
            )
        if unknown:
            raise InspectorError(f"Unknown preset '{unknown[0]}' for model '{model}'.")
        if not selected:
            raise InspectorError("Training requires at least one selected preset.")
        return [name for name, _preset in selected]

    def _field_maps(
        self,
        model: str,
        preset: str,
    ) -> tuple[tuple[ConfigurationField, ...], dict[str, ConfigurationField]]:
        fields = WorkbenchInspectionAdapter.select(model).configuration(
            preset
        ).fields
        by_key: dict[str, ConfigurationField] = {}
        for config_field in fields:
            by_key[normalize_key(config_field.key)] = config_field
        for config_field in fields:
            by_key[
                normalize_key(config_key_to_model_param(config_field.key))
            ] = by_key.get(
                normalize_key(config_key_to_model_param(config_field.key)),
                config_field,
            )
        return fields, by_key

    def _training_command(
        self,
        *,
        model: str,
        preset: str,
        experiment_task: str,
        dataset: str,
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str],
    ) -> str:
        fields, by_key = self._field_maps(model, preset)
        return _build_training_command(
            fields=fields,
            by_key=by_key,
            model=model,
            preset=preset,
            experiment_task=experiment_task,
            dataset=dataset,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
        )

    def _total_epochs(self, parts, parsed_overrides: dict[str, Any]) -> int:
        raw_epochs = parsed_overrides.get(
            "num_epochs",
            getattr(parts.runtime_defaults, "NUM_EPOCHS", 10),
        )
        try:
            return max(0, int(raw_epochs))
        except (TypeError, ValueError):
            return 0

    def _run_total_epochs(self, parts: ModelPackage, run: RunSpec) -> int:
        parsed_overrides = WorkbenchInspectionAdapter.from_package(
            parts
        ).parse_overrides(dict(run.overrides)).values
        return self._total_epochs(parts, dict(parsed_overrides))

    def _pending_run(
        self,
        *,
        model: str,
        index: int,
        preset: str,
        experiment_task: str,
        dataset: str,
        changes: list[dict[str, Any]],
        overrides: dict[str, Any],
        log_folder: str,
        monitors: list[str],
        total_epochs: int,
    ) -> dict[str, Any]:
        return {
            "id": f"run-{index:04d}",
            "index": index,
            "status": "Pending",
            "preset": preset,
            "experimentTask": experiment_task,
            "dataset": dataset,
            "changes": changes,
            "overrides": overrides,
            "command": self._training_command(
                model=model,
                preset=preset,
                experiment_task=experiment_task,
                dataset=dataset,
                overrides=overrides,
                log_folder=log_folder,
                monitors=monitors,
            ),
            "totalEpochs": total_epochs,
            "currentEpoch": 0,
            "metrics": {},
            "logDir": None,
            "error": None,
            "errorTraceback": None,
        }

    def _plan_payload(
        self,
        *,
        model: str,
        semantic_plan: RunPlan,
        log_folder: str,
        runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            **model_identity_payload_from_id(model),
            "preset": semantic_plan.presets[0],
            "presets": list(semantic_plan.presets),
            "experimentTask": semantic_plan.experiment_task,
            "datasets": list(semantic_plan.datasets),
            "overrides": dict(semantic_plan.overrides),
            "search": _workbench_search_payload(semantic_plan.search),
            "logFolder": log_folder,
            "isRandomSearch": bool(
                semantic_plan.search and semantic_plan.search.mode == "random"
            ),
            "runs": runs,
            "summary": self.summarize(runs),
        }

    def create_run_plan(
        self,
        command: CreateTrainingRunPlanCommand,
    ) -> TrainingRunPlanView:
        payload = self.create_for_request(
            model=command.model,
            preset=command.preset,
            presets=command.presets,
            experiment_task=command.experiment_task,
            datasets=command.datasets,
            overrides=command.overrides,
            log_folder=command.log_folder,
            monitors=command.monitors,
            search=(
                training_search_to_payload(command.search)
                if command.search is not None
                else None
            ),
        )
        return training_run_plan_from_payload(payload)

    def materialize_training_job(
        self,
        command: CreateTrainingJobCommand,
        *,
        validated_log_folder: str,
    ) -> MaterializedTrainingRunPlan:
        """Accept one job submission and derive its canonical persisted plan."""

        selected = self.resolve_inputs(
            model=command.model,
            preset=command.preset,
            presets=command.presets,
            experiment_task=command.experiment_task,
            datasets=command.datasets,
            overrides=command.overrides,
            search=(
                training_search_to_payload(command.search)
                if command.search is not None
                else None
            ),
        )
        monitor_names = self.resolve_monitor_names(
            selected.parts,
            command.monitors,
        )
        document = (
            self.from_submitted(
                model=command.model,
                selected=selected,
                run_plan=command.run_plan,
                log_folder=validated_log_folder,
                monitors=monitor_names,
            )
            if command.run_plan is not None
            else self.create(
                model=command.model,
                selected=selected,
                log_folder=validated_log_folder,
                monitors=monitor_names,
            )
        )
        return MaterializedTrainingRunPlan(
            document=encode_persisted_run_plan(document),
            monitors=tuple(monitor_names),
        )

    def accept_worker_payload(
        self,
        package: ModelPackage,
        payload: Mapping[str, Any],
    ) -> RunPlan:
        """Defensively reconstruct and validate one persisted exact Run Plan."""

        rows = _worker_plan_rows(payload)
        persisted_plan = _validate_worker_plan_envelope(
            package.catalog_key,
            payload,
        )
        _datasets, monitors = worker_payload_names(payload)
        request = _worker_run_request(persisted_plan)
        _validate_worker_authoritative_metadata(persisted_plan, rows)
        submitted_runs = _worker_submitted_runs(persisted_plan, rows)
        try:
            semantic_plan = accept_run_plan(
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

        expected_rows: list[dict[str, Any]] = []
        for index, (raw_row, semantic_run) in enumerate(
            zip(rows, semantic_plan.runs, strict=True),
            start=1,
        ):
            assert isinstance(raw_row, Mapping)
            expected_row = self._pending_semantic_run(
                model=package.catalog_key,
                parts=package,
                run=semantic_run,
                index=index,
                log_folder=str(persisted_plan["logFolder"]),
                monitors=monitors,
                search=semantic_plan.search,
            )
            for field_name in ("snapshotId", "snapshotName"):
                if field_name in raw_row:
                    raw_value = raw_row.get(field_name)
                    if raw_value is not None and not isinstance(raw_value, str):
                        raise ValueError(
                            f"Run plan row {index} {field_name} must be a string "
                            "or null."
                        )
                    expected_row[field_name] = raw_value
            raw_total_epochs = raw_row.get("totalEpochs")
            if (
                isinstance(raw_total_epochs, bool)
                or not isinstance(raw_total_epochs, int)
                or raw_total_epochs != expected_row["totalEpochs"]
            ):
                raise ValueError(
                    f"Run plan row {index} total epochs do not match its "
                    "accepted Runtime Defaults."
                )
            if dict(raw_row) != expected_row:
                differing_fields = sorted(
                    field_name
                    for field_name in set(raw_row) | set(expected_row)
                    if raw_row.get(field_name) != expected_row.get(field_name)
                    or (field_name in raw_row) != (field_name in expected_row)
                )
                if differing_fields == ["command"]:
                    raise ValueError(
                        "Run plan log folder or monitors do not match its "
                        f"row {index} command."
                    )
                raise ValueError(
                    f"Run plan row {index} projection does not match its "
                    f"accepted Run: {', '.join(differing_fields)}."
                )
            expected_rows.append(expected_row)

        expected_plan = self._plan_payload(
            model=package.catalog_key,
            semantic_plan=semantic_plan,
            log_folder=str(persisted_plan["logFolder"]),
            runs=expected_rows,
        )
        if persisted_plan.get("summary") != expected_plan["summary"]:
            raise ValueError(
                "Run plan summary does not match its accepted Run rows."
            )
        if dict(persisted_plan) != expected_plan:
            raise ValueError(
                "Persisted Run plan projection does not match its accepted Runs."
            )
        planned_run_count = payload.get("plannedRunCount")
        if (
            isinstance(planned_run_count, bool)
            or not isinstance(planned_run_count, int)
            or planned_run_count != len(expected_rows)
        ):
            raise ValueError(
                "Training payload planned run count does not match its Run Plan."
            )
        return semantic_plan


def accept_worker_run_plan(
    package: ModelPackage,
    payload: Mapping[str, Any],
) -> RunPlan:
    return WorkbenchRunPlanAdapter().accept_worker_payload(package, payload)


__all__ = [
    "MaterializedTrainingRunPlan",
    "SubmittedTrainingRun",
    "SubmittedTrainingRunPlan",
    "SelectedTrainingInputs",
    "TrainingRunPlanSummaryView",
    "TrainingRunPlanView",
    "TrainingRunView",
    "TrainingSearch",
    "WorkbenchRunPlanAdapter",
    "accept_worker_run_plan",
    "submitted_run_plan_from_payload",
    "submitted_run_plan_to_payload",
    "training_run_plan_from_payload",
    "training_run_plan_to_payload",
    "training_search_from_payload",
    "training_search_to_payload",
    "worker_payload_names",
]
