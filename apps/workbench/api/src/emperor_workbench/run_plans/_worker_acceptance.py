from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_runtime.packages import normalize_key
from model_runtime.runs import (
    PlanningBudget,
    RunPlan,
    RunRequest,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)

from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageIdentity,
)
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
    ProjectAdapterFailure,
)
from emperor_workbench.run_plans._command_projection import project_pending_run
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_DATASETS,
    MAX_TRAINING_MONITORS,
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._persistence_codec import (
    _plan_to_payload,
    _run_from_payload,
    _run_to_payload,
    _snapshot_revisions_from_payload,
    _validate_config_mapping,
    _validate_config_value,
    _validate_run_payload,
    _validate_summary_payload,
)
from emperor_workbench.run_plans._progress_projection import (
    RunPlanProgressProjector,
)
from emperor_workbench.run_plans._records import TrainingRunPlanView
from emperor_workbench.run_plans._search import search_from_spec


def _bounded_names(
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
        raise ValueError(f"Training worker payload accepts at most {limit} {field}.")
    if any(not isinstance(name, str) or not name.strip() for name in raw_names):
        raise ValueError(
            f"Training worker payload {field} must contain non-empty strings."
        )
    if len(raw_names) != len(set(raw_names)):
        raise ValueError(
            f"Training worker payload {field} must not contain duplicate names."
        )
    return list(raw_names)


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
            f"Training search accepts at most {MAX_TRAINING_SEARCH_AXES} selected axes."
        )
    mode = raw_search.get("mode")
    if mode not in {"grid", "random"}:
        raise ValueError("Training search mode must be 'grid' or 'random'.")
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
    if mode == "grid" and random_samples is not None:
        raise ValueError("Grid search must not include a random sample count.")
    axes: list[SearchAxisSelection] = []
    normalized_axes: set[str] = set()
    for key, values in raw_values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Training search axes must have non-empty string names.")
        normalized_axis = normalize_key(key)
        if normalized_axis in normalized_axes:
            raise ValueError(f"Training search contains duplicate axis '{key}'.")
        normalized_axes.add(normalized_axis)
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search axis '{key}' requires at least one selected value."
            )
        if len(values) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise ValueError(
                f"Search axis '{key}' accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} selected values."
            )
        for index, value in enumerate(values):
            _validate_config_value(
                value,
                path=f"Training search axis '{key}' value {index + 1}",
            )
        axes.append(SearchAxisSelection(key=key, values=tuple(values)))
    return SearchSpec(
        mode=mode,
        axes=tuple(axes),
        random_samples=random_samples,
    )


def _run_request(run_plan: Mapping[str, Any]) -> RunRequest:
    raw_presets = run_plan.get("presets") or [run_plan.get("preset")]
    raw_datasets = run_plan.get("datasets")
    if not isinstance(raw_presets, list):
        raise ValueError("Training payload presets must be a list.")
    if not isinstance(raw_datasets, list):
        raise ValueError("Training payload datasets must be a list.")
    raw_overrides = run_plan.get("overrides") or {}
    _validate_config_mapping(
        raw_overrides,
        path="Training payload overrides",
    )
    experiment_task = run_plan.get("experimentTask")
    return RunRequest(
        presets=tuple(str(preset) for preset in raw_presets if preset is not None),
        datasets=tuple(str(dataset) for dataset in raw_datasets),
        experiment_task=(str(experiment_task) if experiment_task is not None else None),
        overrides=dict(raw_overrides),
        search=_search_spec(run_plan.get("search")),
    )


def _plan_rows(payload: Mapping[str, Any]) -> list[Any]:
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


def _submitted_runs(
    run_plan: Mapping[str, Any],
    rows: list[Any],
) -> tuple[SubmittedRun, ...]:
    plan_task = run_plan.get("experimentTask") or None
    submitted: list[SubmittedRun] = []
    seen_run_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Run plan row {index} must be an object.")
        _validate_run_payload(row, path=f"Run plan row {index}")
        run_id = row.get("id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"Run plan row {index} requires a non-empty id.")
        if run_id in seen_run_ids:
            raise ValueError(f"Run plan contains duplicate run id '{run_id}'.")
        seen_run_ids.add(run_id)
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
        _validate_config_mapping(
            overrides,
            path=f"Run plan row {index} overrides",
        )
        submitted.append(
            SubmittedRun(
                id=run_id,
                preset=preset,
                dataset=dataset,
                overrides=dict(overrides),
            )
        )
    return tuple(submitted)


def _validate_envelope(
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
        isinstance(preset, str) and preset.strip() for preset in run_plan["presets"]
    ):
        raise ValueError("Run plan presets must contain non-empty strings.")
    if len(run_plan["presets"]) != len(set(run_plan["presets"])):
        raise ValueError("Run plan presets must not contain duplicate names.")
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
        isinstance(dataset, str) and dataset.strip() for dataset in run_plan["datasets"]
    ):
        raise ValueError("Run plan datasets must contain non-empty strings.")
    if len(run_plan["datasets"]) != len(set(run_plan["datasets"])):
        raise ValueError("Run plan datasets must not contain duplicate names.")
    _validate_config_mapping(
        run_plan["overrides"],
        path="Run plan overrides",
    )
    if run_plan["search"] is not None and not isinstance(run_plan["search"], Mapping):
        raise ValueError("Run plan search must be an object or null.")
    if not isinstance(run_plan["logFolder"], str):
        raise ValueError("Run plan log folder must be a string.")
    plan_identity = ModelPackageIdentity.from_mapping(run_plan)
    if plan_identity is None:
        raise ValueError("Run plan does not include a valid model identity.")
    plan_model_id = plan_identity.catalog_key
    if plan_model_id != model_id:
        raise ValueError(
            f"Run plan model '{plan_model_id}' does not match selected model "
            f"'{model_id}'."
        )
    return run_plan


def _validate_authoritative_metadata(
    run_plan: Mapping[str, Any],
    rows: list[Any],
) -> None:
    presets = set(run_plan["presets"])
    datasets = set(run_plan["datasets"])
    envelope_overrides = dict(run_plan["overrides"])
    raw_search = run_plan["search"]
    search_values = (
        dict(raw_search.get("values") or {}) if isinstance(raw_search, Mapping) else {}
    )
    expected_random = bool(
        isinstance(raw_search, Mapping) and raw_search.get("mode") == "random"
    )
    snapshot_revisions = _snapshot_revisions_from_payload(
        run_plan.get("snapshotRevisions")
    )
    revision_ids = {revision.id for revision in snapshot_revisions}
    row_snapshot_ids = {
        str(row.get("snapshotId"))
        for row in rows
        if isinstance(row, Mapping) and row.get("snapshotId") is not None
    }
    if row_snapshot_ids != revision_ids:
        raise ValueError(
            "Run plan Snapshot rows do not match its semantic revision set."
        )
    if run_plan.get("isRandomSearch") is not expected_random:
        raise ValueError(
            "Run plan random-search marker does not match its Search Metadata."
        )

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        if row.get("preset") not in presets:
            raise ValueError(f"Run plan presets do not contain row {index} preset.")
        if row.get("dataset") not in datasets:
            raise ValueError(f"Run plan datasets do not contain row {index} dataset.")
        row_overrides = row.get("overrides")
        if not isinstance(row_overrides, Mapping):
            continue
        normalized_row_overrides = {
            normalize_key(str(key)): value for key, value in row_overrides.items()
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


def _planning_budget() -> PlanningBudget:
    return PlanningBudget(
        max_axes=MAX_TRAINING_SEARCH_AXES,
        max_values_per_axis=MAX_TRAINING_SEARCH_AXIS_VALUES,
        max_materialized_runs=MAX_TRAINING_PLANNED_RUNS,
    )


def _expected_plan(
    *,
    model: str,
    semantic_plan: RunPlan,
    log_folder: str,
    rows: list[dict[str, Any]],
    snapshot_revisions,
) -> TrainingRunPlanView:
    runs = [_run_from_payload(row) for row in rows]
    return TrainingRunPlanView(
        model=model,
        preset=semantic_plan.presets[0],
        presets=list(semantic_plan.presets),
        experiment_task=semantic_plan.experiment_task,
        datasets=list(semantic_plan.datasets),
        overrides=dict(semantic_plan.overrides),
        search=search_from_spec(semantic_plan.search),
        log_folder=log_folder,
        is_random_search=bool(
            semantic_plan.search and semantic_plan.search.mode == "random"
        ),
        runs=runs,
        summary=RunPlanProgressProjector.summarize(runs),
        snapshot_revisions=snapshot_revisions,
    )


@dataclass(frozen=True, slots=True)
class TrainingWorkerPlanContext:
    model_id: str
    model_type: str
    model_name: str
    preset: str
    presets: tuple[str, ...]
    experiment_task: str
    datasets: tuple[str, ...]
    monitors: tuple[str, ...]
    log_folder: str


class RunPlanWorkerAcceptance:
    """Defensively reconstruct one exact persisted Run Plan for a worker."""

    @staticmethod
    def payload_names(
        payload: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        raw_plan = payload.get("runPlan")
        datasets = (
            _bounded_names(
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
            _bounded_names(
                payload,
                "monitors",
                limit=MAX_TRAINING_MONITORS,
                required=False,
            ),
        )

    @staticmethod
    def describe(payload: Mapping[str, Any]) -> TrainingWorkerPlanContext:
        raw_plan = payload.get("runPlan")
        if not isinstance(raw_plan, Mapping):
            raise ValueError(
                "Training payload does not include a non-empty materialized run plan."
            )
        identity = ModelPackageIdentity.from_mapping(raw_plan)
        if identity is None:
            raise ValueError(
                "Training payload does not include a valid model identity."
            )
        persisted_plan = _validate_envelope(identity.catalog_key, payload)
        datasets, monitors = RunPlanWorkerAcceptance.payload_names(payload)
        return TrainingWorkerPlanContext(
            model_id=identity.catalog_key,
            model_type=identity.model_type,
            model_name=identity.model,
            preset=str(persisted_plan["preset"]),
            presets=tuple(str(value) for value in persisted_plan["presets"]),
            experiment_task=str(persisted_plan["experimentTask"]),
            datasets=tuple(datasets),
            monitors=tuple(monitors),
            log_folder=str(persisted_plan["logFolder"]),
        )

    @staticmethod
    def accept(
        package: ModelPackageReference,
        payload: Mapping[str, Any],
    ) -> RunPlan:
        rows = _plan_rows(payload)
        persisted_plan = _validate_envelope(package.catalog_key, payload)
        _validate_summary_payload(persisted_plan.get("summary"))
        _datasets, monitors = RunPlanWorkerAcceptance.payload_names(payload)
        request = _run_request(persisted_plan)
        _validate_authoritative_metadata(persisted_plan, rows)
        submitted_runs = _submitted_runs(persisted_plan, rows)
        try:
            semantic_plan = package.client.accept_run_plan(
                package.catalog_key,
                request,
                submitted_runs,
                budget=_planning_budget(),
            )
        except ProjectAdapterFailure as exc:
            raise ValueError(exc.detail) from exc

        model_packages = ModelPackageCatalog(package.client)
        expected_rows: list[dict[str, Any]] = []
        for index, (raw_row, semantic_run) in enumerate(
            zip(rows, semantic_plan.runs, strict=True),
            start=1,
        ):
            assert isinstance(raw_row, Mapping)
            expected_row = _run_to_payload(
                project_pending_run(
                    model_packages,
                    model=package.catalog_key,
                    package=package,
                    run=semantic_run,
                    index=index,
                    log_folder=str(persisted_plan["logFolder"]),
                    monitors=monitors,
                    search=semantic_plan.search,
                )
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
                command_projection_fields = {
                    "commandArgv",
                    "commands",
                }
                if set(differing_fields).issubset(command_projection_fields):
                    raise ValueError(
                        "Run plan log folder or monitors do not match its "
                        f"row {index} command."
                    )
                raise ValueError(
                    f"Run plan row {index} projection does not match its "
                    f"accepted Run: {', '.join(differing_fields)}."
                )
            expected_rows.append(expected_row)

        expected_plan = _plan_to_payload(
            _expected_plan(
                model=package.catalog_key,
                semantic_plan=semantic_plan,
                log_folder=str(persisted_plan["logFolder"]),
                rows=expected_rows,
                snapshot_revisions=_snapshot_revisions_from_payload(
                    persisted_plan.get("snapshotRevisions")
                ),
            )
        )
        if "snapshotRevisions" not in persisted_plan:
            expected_plan.pop("snapshotRevisions", None)
        if persisted_plan.get("summary") != expected_plan["summary"]:
            raise ValueError("Run plan summary does not match its accepted Run rows.")
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

    @staticmethod
    def execute(
        payload: Mapping[str, Any],
        *,
        logs_root: Path,
        progress_path: Path,
        progress_step_interval: int,
    ) -> RunPlan:
        context = RunPlanWorkerAcceptance.describe(payload)
        with ProjectAdapterClient(timeout_seconds=None) as project_adapter:
            package = project_adapter.package(context.model_id)
            plan = RunPlanWorkerAcceptance.accept(package, payload)
            package.client.execute_run_plan(
                package.catalog_key,
                plan,
                logs_root=str(logs_root),
                log_folder=context.log_folder or None,
                progress_path=str(progress_path),
                progress_step_interval=progress_step_interval,
                monitors=list(context.monitors),
            )
        return plan


__all__ = ["RunPlanWorkerAcceptance", "TrainingWorkerPlanContext"]
