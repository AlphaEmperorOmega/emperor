from __future__ import annotations

import copy
from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal, cast

from model_runtime.packages import normalize_key

from emperor_workbench.model_packages import ModelPackageIdentity
from emperor_workbench.run_plans._limits import (
    MAX_TRAINING_PLANNED_RUNS,
    MAX_TRAINING_SEARCH_AXES,
    MAX_TRAINING_SEARCH_AXIS_VALUES,
)
from emperor_workbench.run_plans._records import (
    ConfigSnapshotRevision,
    TrainingCommandsView,
    TrainingRunChangeSource,
    TrainingRunChangeView,
    TrainingRunPlanSummaryView,
    TrainingRunPlanView,
    TrainingRunStatus,
    TrainingRunView,
    TrainingSearch,
)

_RUN_STATUSES = {
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Cancelled",
    "Skipped",
}
_CHANGE_SOURCES = {"override", "search"}


def _validate_config_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError(f"{path} must be finite.")
    raise ValueError(f"{path} must be a scalar config value.")


def _validate_config_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object.")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} must use string keys.")
        _validate_config_value(item, path=f"{path}.{key}")
    return value


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError(f"{path} must be finite.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} must use string keys.")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} must be JSON-compatible.")


def _validate_string_list(
    value: object,
    *,
    path: str,
    allow_empty: bool = True,
    nonempty_items: bool = False,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        requirement = "a non-empty list" if not allow_empty else "a list"
        raise ValueError(f"{path} must be {requirement}.")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} must contain strings.")
    if nonempty_items and any(not item for item in value):
        raise ValueError(f"{path} must contain non-empty strings.")
    if unique and len(value) != len(set(value)):
        raise ValueError(f"{path} must not contain duplicate names.")
    return value


def _validate_integer(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")


def _validate_search_payload(
    payload: object,
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    mode = payload.get("mode")
    if mode not in {"grid", "random"}:
        raise ValueError(f"{path}.mode must be 'grid' or 'random'.")
    values = payload.get("values")
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{path}.values must be a non-empty object.")
    if len(values) > MAX_TRAINING_SEARCH_AXES:
        raise ValueError(
            f"{path}.values accepts at most {MAX_TRAINING_SEARCH_AXES} axes."
        )
    normalized_axes: set[str] = set()
    for key, options in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{path}.values must use non-empty string keys.")
        normalized_key = normalize_key(key)
        if normalized_key in normalized_axes:
            raise ValueError(f"{path}.values contains duplicate axis '{key}'.")
        normalized_axes.add(normalized_key)
        if not isinstance(options, list) or not options:
            raise ValueError(f"{path}.values.{key} must be a non-empty list.")
        if len(options) > MAX_TRAINING_SEARCH_AXIS_VALUES:
            raise ValueError(
                f"{path}.values.{key} accepts at most "
                f"{MAX_TRAINING_SEARCH_AXIS_VALUES} values."
            )
        for index, option in enumerate(options):
            _validate_config_value(
                option,
                path=f"{path}.values.{key}[{index}]",
            )
    random_samples = payload.get("randomSamples")
    if mode == "random":
        _validate_integer(random_samples, path=f"{path}.randomSamples")
        assert isinstance(random_samples, int)
        if not 1 <= random_samples <= MAX_TRAINING_PLANNED_RUNS:
            raise ValueError(
                f"{path}.randomSamples must be between 1 and "
                f"{MAX_TRAINING_PLANNED_RUNS}."
            )
    elif random_samples is not None:
        raise ValueError(f"{path}.randomSamples is only valid for random search.")
    return payload


def _validate_run_change(payload: object, *, path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    for field in ("key", "label"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{path}.{field} must be a string.")
    _validate_config_value(payload.get("value"), path=f"{path}.value")
    if payload.get("source") not in _CHANGE_SOURCES:
        raise ValueError(f"{path}.source is invalid.")


def _validate_run_payload(payload: object, *, path: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    for field in ("id", "preset", "dataset"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}.{field} must be a non-empty string.")
    for field in (
        "experimentTask",
        "command",
        "logDir",
        "error",
        "errorTraceback",
        "snapshotId",
        "snapshotName",
    ):
        if (
            field in payload
            and payload[field] is not None
            and not isinstance(payload[field], str)
        ):
            raise ValueError(f"{path}.{field} must be a string or null.")
    for field in ("index", "totalEpochs", "currentEpoch"):
        if field in payload:
            _validate_integer(payload[field], path=f"{path}.{field}")
    if "status" in payload and payload["status"] not in _RUN_STATUSES:
        raise ValueError(f"{path}.status is invalid.")
    if "changes" in payload:
        changes = payload["changes"]
        if not isinstance(changes, list):
            raise ValueError(f"{path}.changes must be a list.")
        for index, change in enumerate(changes):
            _validate_run_change(change, path=f"{path}.changes[{index}]")
    if "overrides" in payload:
        _validate_config_mapping(payload["overrides"], path=f"{path}.overrides")
    if "commandArgv" in payload:
        _validate_string_list(payload["commandArgv"], path=f"{path}.commandArgv")
    if "commands" in payload:
        commands = payload["commands"]
        if not isinstance(commands, Mapping):
            raise ValueError(f"{path}.commands must be an object.")
        for shell in ("posix", "powershell"):
            if not isinstance(commands.get(shell), str):
                raise ValueError(f"{path}.commands.{shell} must be a string.")
    if "metrics" in payload:
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{path}.metrics must be an object.")
        _validate_json_value(metrics, path=f"{path}.metrics")


def _validate_summary_payload(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Persisted Run Plan summary must be an object.")
    for field in (
        "totalRuns",
        "completedRuns",
        "runningRuns",
        "pendingRuns",
        "failedRuns",
        "cancelledRuns",
        "skippedRuns",
        "totalEpochs",
        "completedEpochs",
        "remainingEpochs",
    ):
        if field in payload:
            _validate_integer(payload[field], path=f"runPlan.summary.{field}")


def _validate_plan_payload(payload: Mapping[str, Any]) -> None:
    if ModelPackageIdentity.from_mapping(payload) is None:
        raise ValueError("Persisted Run Plan model identity is invalid.")
    preset = payload.get("preset")
    if not isinstance(preset, str) or not preset:
        raise ValueError("Persisted Run Plan preset must be a non-empty string.")
    _validate_string_list(
        payload.get("presets"),
        path="runPlan.presets",
        allow_empty=False,
        nonempty_items=True,
        unique=True,
    )
    _validate_string_list(
        payload.get("datasets"),
        path="runPlan.datasets",
        allow_empty=False,
        nonempty_items=True,
        unique=True,
    )
    if "experimentTask" in payload and not isinstance(payload["experimentTask"], str):
        raise ValueError("runPlan.experimentTask must be a string.")
    _validate_config_mapping(payload.get("overrides"), path="runPlan.overrides")
    if payload.get("search") is not None:
        _validate_search_payload(payload["search"], path="runPlan.search")
    if not isinstance(payload.get("logFolder"), str):
        raise ValueError("runPlan.logFolder must be a string.")
    if "isRandomSearch" in payload and not isinstance(payload["isRandomSearch"], bool):
        raise ValueError("runPlan.isRandomSearch must be a boolean.")
    rows = payload.get("runs")
    if not isinstance(rows, list):
        raise ValueError("Persisted Run Plan rows must be a list of objects.")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        _validate_run_payload(row, path=f"runPlan.runs[{index}]")
        assert isinstance(row, Mapping)
        run_id = cast(str, row["id"])
        if run_id in seen_ids:
            raise ValueError(
                f"Persisted Run Plan contains duplicate run id '{run_id}'."
            )
        seen_ids.add(run_id)
    _validate_summary_payload(payload.get("summary"))
    _snapshot_revisions_from_payload(payload.get("snapshotRevisions"))


def _mapping_items(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _payload_model_id(payload: Mapping[str, Any]) -> str:
    identity = ModelPackageIdentity.from_mapping(payload)
    if identity is not None:
        return identity.catalog_key
    return str(payload.get("model") or "")


def _model_identity(model_id: str) -> dict[str, str]:
    identity = ModelPackageIdentity.from_id(model_id)
    return {"modelType": identity.model_type, "model": identity.model}


def _search_from_payload(
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


def _search_to_payload(search: TrainingSearch) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": search.mode, "values": search.values}
    if search.random_samples is not None:
        payload["randomSamples"] = search.random_samples
    return payload


def _run_change_from_payload(
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


def _run_change_to_payload(change: TrainingRunChangeView) -> dict[str, Any]:
    return {
        "key": change.key,
        "label": change.label,
        "value": change.value,
        "source": change.source,
    }


def _run_from_payload(payload: Mapping[str, Any]) -> TrainingRunView:
    snapshot_id = payload.get("snapshotId")
    snapshot_name = payload.get("snapshotName")
    log_dir = payload.get("logDir")
    error = payload.get("error")
    error_traceback = payload.get("errorTraceback")
    command = str(payload.get("command") or "")
    raw_commands = payload.get("commands")
    commands = raw_commands if isinstance(raw_commands, Mapping) else {}
    raw_command_argv = payload.get("commandArgv")
    return TrainingRunView(
        id=str(payload.get("id") or ""),
        index=int(payload.get("index") or 0),
        status=cast(TrainingRunStatus, str(payload.get("status") or "Pending")),
        preset=str(payload.get("preset") or ""),
        dataset=str(payload.get("dataset") or ""),
        experiment_task=str(payload.get("experimentTask") or ""),
        changes=[
            _run_change_from_payload(item)
            for item in _mapping_items(payload.get("changes"))
        ],
        overrides=dict(payload.get("overrides") or {}),
        command=command,
        command_argv=(
            [str(value) for value in raw_command_argv]
            if isinstance(raw_command_argv, list)
            else []
        ),
        commands=TrainingCommandsView(
            posix=str(commands.get("posix") or command),
            powershell=str(commands.get("powershell") or command),
        ),
        total_epochs=int(payload.get("totalEpochs") or 0),
        snapshot_id=str(snapshot_id) if snapshot_id is not None else None,
        snapshot_name=str(snapshot_name) if snapshot_name is not None else None,
        snapshot_id_present="snapshotId" in payload,
        snapshot_name_present="snapshotName" in payload,
        current_epoch=int(payload.get("currentEpoch") or 0),
        metrics=dict(payload.get("metrics") or {}),
        log_dir=str(log_dir) if log_dir is not None else None,
        error=str(error) if error is not None else None,
        error_traceback=(str(error_traceback) if error_traceback is not None else None),
    )


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


def _summary_from_payload(
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


def _snapshot_revisions_from_payload(
    payload: object,
) -> tuple[ConfigSnapshotRevision, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError("Snapshot revisions must be a list.")
    revisions: list[ConfigSnapshotRevision] = []
    seen: set[str] = set()
    for raw_revision in payload:
        if not isinstance(raw_revision, Mapping):
            raise ValueError("Snapshot revisions must contain objects.")
        snapshot_id = raw_revision.get("id")
        semantic_revision = raw_revision.get("semanticRevision")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("Snapshot revision id must be non-empty.")
        if (
            not isinstance(semantic_revision, str)
            or len(semantic_revision) != 64
            or any(
                character not in "0123456789abcdef" for character in semantic_revision
            )
        ):
            raise ValueError(f"Snapshot '{snapshot_id}' semantic revision is invalid.")
        if snapshot_id in seen:
            raise ValueError(f"Duplicate snapshot revision '{snapshot_id}'.")
        seen.add(snapshot_id)
        revisions.append(
            ConfigSnapshotRevision(
                id=snapshot_id,
                semantic_revision=semantic_revision,
            )
        )
    return tuple(revisions)


def _snapshot_revisions_to_payload(
    revisions: tuple[ConfigSnapshotRevision, ...],
) -> list[dict[str, str]]:
    return [
        {"id": revision.id, "semanticRevision": revision.semantic_revision}
        for revision in revisions
    ]


def _plan_from_payload(payload: Mapping[str, Any]) -> TrainingRunPlanView:
    return TrainingRunPlanView(
        model=_payload_model_id(payload),
        preset=str(payload.get("preset") or ""),
        presets=[str(item) for item in payload.get("presets") or []],
        experiment_task=str(payload.get("experimentTask") or ""),
        datasets=[str(item) for item in payload.get("datasets") or []],
        overrides=dict(payload.get("overrides") or {}),
        search=_search_from_payload(
            cast(Mapping[str, Any] | None, payload.get("search"))
        ),
        log_folder=str(payload.get("logFolder") or ""),
        is_random_search=bool(payload.get("isRandomSearch")),
        runs=[_run_from_payload(item) for item in _mapping_items(payload.get("runs"))],
        summary=_summary_from_payload(
            cast(Mapping[str, Any] | None, payload.get("summary"))
        ),
        snapshot_revisions=_snapshot_revisions_from_payload(
            payload.get("snapshotRevisions")
        ),
    )


def _plan_to_payload(plan: TrainingRunPlanView) -> dict[str, Any]:
    return {
        **_model_identity(plan.model),
        "preset": plan.preset,
        "presets": plan.presets,
        "experimentTask": plan.experiment_task,
        "datasets": plan.datasets,
        "overrides": plan.overrides,
        "search": _search_to_payload(plan.search) if plan.search else None,
        "logFolder": plan.log_folder,
        "isRandomSearch": plan.is_random_search,
        "runs": [_run_to_payload(run) for run in plan.runs],
        "summary": _summary_to_payload(plan.summary),
        "snapshotRevisions": _snapshot_revisions_to_payload(plan.snapshot_revisions),
    }


class RunPlanPersistenceCodec:
    """Own the stable persisted and worker JSON representation of a Run Plan."""

    @staticmethod
    def decode(payload: object) -> TrainingRunPlanView:
        if not isinstance(payload, Mapping):
            raise ValueError("Persisted Run Plan must be an object.")
        _validate_plan_payload(payload)
        return _plan_from_payload(payload)

    @staticmethod
    def encode(plan: TrainingRunPlanView) -> dict[str, Any]:
        return copy.deepcopy(_plan_to_payload(plan))

    @staticmethod
    def decode_search(payload: Mapping[str, Any] | None) -> TrainingSearch | None:
        if payload is not None:
            _validate_search_payload(payload, path="search")
        return _search_from_payload(payload)

    @staticmethod
    def encode_search(search: TrainingSearch) -> dict[str, Any]:
        return copy.deepcopy(_search_to_payload(search))


__all__ = ["RunPlanPersistenceCodec"]
