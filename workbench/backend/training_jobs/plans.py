"""Adapt Workbench Training inputs to authoritative Emperor Run plans."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from emperor.inspection import (
    InspectionError,
    SearchAxis,
    configuration_schema,
    parse_overrides,
    reject_locked_overrides,
    resolve_override_key,
    search_space_schema,
    serialize_overrides,
)
from emperor.model_packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_model_param,
    dataset_name,
    iter_supported_config_keys,
    model_identity_payload_from_id,
    model_package,
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

from workbench.backend.inspection_errors import call_inspection
from workbench.backend.inspection_serialization import (
    configuration_schema_payload,
)
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import is_valid_log_experiment_name
from workbench.backend.training_jobs.contracts import (
    CreateTrainingRunPlanCommand,
    TrainingRunPlanView,
)
from workbench.backend.training_jobs.limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from workbench.backend.training_jobs.serialization import (
    training_run_plan_from_payload,
    training_search_to_payload,
)


def _require_package(model: str) -> ModelPackage:
    package = model_package(model)
    if package is None:
        raise InspectorError(f"Unknown model: {model}")
    return package


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

    semantic_search = call_inspection(
        search_space_schema,
        package,
        preset_name,
    )
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


def _build_training_command(
    *,
    fields: list[dict[str, Any]],
    by_key: dict[str, dict[str, Any]],
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
            values_by_field_key[str(field["key"])] = raw_value

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
        field_key = str(field["key"])
        if field_key not in values_by_field_key:
            continue
        config_parts.extend(
            [
                str(field["flag"]),
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
    total_epochs: int


class TrainingRunPlanBuilder:
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
        parsed_overrides = self._parse_and_validate_overrides(
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
            total_epochs=self._total_epochs(parts, parsed_overrides),
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
        parsed_overrides = call_inspection(
            parse_overrides,
            parts,
            effective_overrides,
        ).values
        for selected_preset in selected_preset_names:
            try:
                reject_locked_overrides(parts, selected_preset, parsed_overrides)
            except InspectionError as exc:
                raise InspectorError(str(exc)) from exc
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
                run=run,
                index=index,
                log_folder=log_folder,
                monitors=monitor_names,
                total_epochs=selected.total_epochs,
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
        run: RunSpec,
        index: int,
        log_folder: str,
        monitors: list[str],
        total_epochs: int,
    ) -> dict[str, Any]:
        _fields, by_key = self._field_maps(model, run.preset)
        changes = []
        overrides: dict[str, Any] = {}
        for parameter in run.parameters:
            field = by_key.get(normalize_key(parameter.key))
            field_key = str(field["key"]) if field is not None else parameter.key
            overrides[field_key] = parameter.value
            changes.append(
                {
                    "key": field_key,
                    "label": str(
                        field.get("label", field_key)
                        if field is not None
                        else field_key
                    ),
                    "value": parameter.value,
                    "source": parameter.source,
                }
            )
        return self._pending_run(
            model=model,
            index=index,
            preset=run.preset,
            experiment_task=run.experiment_task,
            dataset=run.dataset,
            changes=changes,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
            total_epochs=total_epochs,
        )

    def from_submitted(
        self,
        *,
        model: str,
        selected: SelectedTrainingInputs,
        run_plan: dict[str, Any],
        log_folder: str,
        monitors: list[str] | None = None,
    ) -> dict[str, Any]:
        monitor_names = monitors or []
        submitted_runs = run_plan.get("runs") or []
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
            row_overrides = dict(row.get("overrides") or {})
            snapshot_id = row.get("snapshotId")
            snapshot_name = row.get("snapshotName")
            canonical_row_overrides = self._canonical_override_values(
                parts=selected.parts,
                overrides=row_overrides,
            )
            runs.append(
                {
                    **row,
                    "id": semantic_run.id,
                    "index": index,
                    "status": "Pending",
                    "preset": semantic_run.preset,
                    "snapshotId": str(snapshot_id) if snapshot_id is not None else None,
                    "snapshotName": str(snapshot_name)
                    if snapshot_name is not None
                    else None,
                    "experimentTask": semantic_plan.experiment_task,
                    "dataset": semantic_run.dataset,
                    "changes": self._canonical_submitted_changes(
                        model=model,
                        preset=semantic_run.preset,
                        changes=list(row.get("changes") or []),
                        overrides=canonical_row_overrides,
                    ),
                    "overrides": canonical_row_overrides,
                    "command": self._training_command(
                        model=model,
                        preset=semantic_run.preset,
                        experiment_task=semantic_plan.experiment_task,
                        dataset=semantic_run.dataset,
                        overrides=canonical_row_overrides,
                        log_folder=log_folder,
                        monitors=monitor_names,
                    ),
                    "totalEpochs": int(row.get("totalEpochs") or 0),
                    "currentEpoch": 0,
                    "metrics": {},
                    "logDir": None,
                    "error": None,
                    "errorTraceback": None,
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
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        package = _require_package(model)
        fields = configuration_schema_payload(
            call_inspection(configuration_schema, package, preset)
        )["fields"]
        by_key: dict[str, dict[str, Any]] = {}
        for field in fields:
            by_key[normalize_key(str(field["key"]))] = field
            by_key[normalize_key(str(field["configKey"]))] = field
        for field in fields:
            by_key[
                normalize_key(config_key_to_model_param(str(field["configKey"])))
            ] = by_key.get(
                normalize_key(config_key_to_model_param(str(field["configKey"]))),
                field,
            )
        return fields, by_key

    def _canonical_override_values(
        self,
        *,
        parts,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return serialize_overrides(parts, overrides)
        except InspectionError as exc:
            raise InspectorError(str(exc)) from exc

    def _canonical_submitted_changes(
        self,
        *,
        model: str,
        preset: str,
        changes: list[dict[str, Any]],
        overrides: dict[str, Any],
    ) -> list[dict[str, Any]]:
        fields, by_key = self._field_maps(model, preset)
        fields_by_key = {str(field["key"]): field for field in fields}
        if not changes:
            return [
                {
                    "key": key,
                    "label": str(fields_by_key.get(key, {}).get("label", key)),
                    "value": value,
                    "source": "override",
                }
                for key, value in overrides.items()
            ]

        canonical_changes: list[dict[str, Any]] = []
        for change in changes:
            raw_key = str(change.get("key") or "")
            field = by_key.get(normalize_key(raw_key))
            key = str(field["key"]) if field is not None else raw_key
            canonical_changes.append(
                {
                    **change,
                    "key": key,
                    "label": str(
                        change.get("label")
                        or (field.get("label") if field is not None else key)
                    ),
                }
            )
        return canonical_changes

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


class TrainingRunPlanService:
    """Typed Workbench Adapter to authoritative Emperor Runs planning."""

    def __init__(self) -> None:
        self._builder = TrainingRunPlanBuilder()

    def create_run_plan(
        self,
        command: CreateTrainingRunPlanCommand,
    ) -> TrainingRunPlanView:
        payload = self._builder.create_for_request(
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


__all__ = [
    "SelectedTrainingInputs",
    "TrainingRunPlanBuilder",
    "TrainingRunPlanService",
]
