from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model_runtime.cli import (
    PROTOCOL_VERSION,
    configuration_schema_to_wire,
    configuration_values_to_wire,
    inspection_result_to_wire,
    json_value_from_wire,
    package_metadata_to_wire,
    planning_budget_from_wire,
    preset_locks_to_wire,
    random_state_from_wire,
    random_state_to_wire,
    run_plan_from_wire,
    run_plan_to_wire,
    run_request_from_wire,
    run_results_to_wire,
    search_space_to_wire,
    submitted_runs_from_wire,
)
from model_runtime.inspection import (
    InspectionError,
    InspectionRequest,
    configuration_schema,
    inspect_model,
    parse_overrides,
    preset_locks,
    reject_locked_overrides,
    search_space_schema,
    serialize_overrides,
    validate_configuration,
)
from model_runtime.packages import (
    abstract_config_class_error,
    parse_config_value,
    serialize_config_value,
)
from model_runtime.runs import (
    FilesystemRunArtifacts,
    JsonlRunProgress,
    RunsError,
    accept_run_plan,
    execute_runs,
    plan_runs,
)
from models.catalog import discover_model_packages, model_package

MAX_ADAPTER_REQUEST_BYTES = 64 * 1024 * 1024


class AdapterProtocolError(ValueError):
    """A request does not conform to the versioned Adapter protocol."""


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterProtocolError(f"{label} must be an object.")
    return value


def _optional_object(
    payload: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    return {} if value is None else _object(value, label)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AdapterProtocolError(f"{label} must be a list.")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AdapterProtocolError(f"{label} must be a string.")
    return value


def _optional_string(value: object, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _strings(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{label} entry") for item in _list(value, label))


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise AdapterProtocolError(f"{label} must be a boolean.")
    return value


def _scalar(value: object, label: str) -> bool | int | float | str | None:
    if (
        value is None
        or type(value) is bool
        or type(value) is int
        or isinstance(value, str)
    ):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise AdapterProtocolError(f"{label} must be a finite JSON scalar.")


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise AdapterProtocolError(f"{label} must be a positive integer.")
    return value


def _tensor_shapes(value: object) -> dict[str, tuple[int, ...]]:
    raw_shapes = _object(value, "tensor_shapes")
    shapes: dict[str, tuple[int, ...]] = {}
    for key, raw_shape in raw_shapes.items():
        if not isinstance(key, str) or not isinstance(raw_shape, list):
            raise AdapterProtocolError(
                "Checkpoint tensor shapes must map names to integer lists."
            )
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in raw_shape
        ):
            raise AdapterProtocolError(
                "Checkpoint tensor shapes must map names to integer lists."
            )
        shapes[key] = tuple(raw_shape)
    return shapes


def _package(payload: Mapping[str, Any]):
    model_id = payload.get("model_id")
    if not isinstance(model_id, str):
        raise AdapterProtocolError("Adapter request requires model_id.")
    package = model_package(model_id)
    if package is None:
        raise ValueError(f"Unknown model: {model_id}")
    return package


def _resolve(payload: Mapping[str, Any]) -> dict[str, Any]:
    package = _package(payload)
    task = package.resolve_experiment_task(
        _optional_string(payload.get("experiment_task"), "experiment_task")
    )
    raw_presets = payload.get("presets")
    raw_datasets = payload.get("datasets")
    raw_monitors = payload.get("monitors")
    preset_names = () if raw_presets is None else _strings(raw_presets, "presets")
    dataset_names = () if raw_datasets is None else _strings(raw_datasets, "datasets")
    monitor_names = () if raw_monitors is None else _strings(raw_monitors, "monitors")
    presets = [package.resolve_preset(name) for name in preset_names]
    datasets = package.resolve_datasets(list(dataset_names), task)
    monitors = package.resolve_monitors(list(monitor_names))
    return {
        "experiment_task": package.task_name(task),
        "presets": [
            {"name": package.preset_name(preset), "key": preset.name}
            for preset in presets
        ],
        "datasets": [dataset.__name__ for dataset in datasets],
        "monitors": [monitor.name for monitor in monitors],
    }


def _handle(operation: str, payload: Mapping[str, Any]) -> Any:
    if operation == "catalog":
        return [
            {
                **package.identity.to_payload(),
                "catalogKey": package.catalog_key,
            }
            for package in discover_model_packages()
        ]
    package = _package(payload)
    if operation == "package_metadata":
        return package_metadata_to_wire(package)
    if operation == "resolve":
        return _resolve(payload)
    if operation == "configuration":
        return configuration_schema_to_wire(
            configuration_schema(
                package,
                _optional_string(payload.get("preset"), "preset"),
            )
        )
    if operation == "search_space":
        raw_presets = payload.get("presets")
        presets = None if raw_presets is None else _strings(raw_presets, "presets")
        return search_space_to_wire(
            search_space_schema(
                package,
                _optional_string(payload.get("preset"), "preset"),
                presets,
            )
        )
    if operation in {"parse_overrides", "serialize_overrides"}:
        overrides = _optional_object(payload, "overrides", "overrides")
        ignore_unknown = _boolean(
            payload.get("ignore_unknown", False),
            "ignore_unknown",
        )
        if operation == "parse_overrides":
            return configuration_values_to_wire(
                parse_overrides(
                    package,
                    overrides,
                    preset=_optional_string(payload.get("preset"), "preset"),
                    ignore_unknown=ignore_unknown,
                ).values
            )
        return configuration_values_to_wire(
            serialize_overrides(
                package,
                overrides,
                ignore_unknown=ignore_unknown,
            )
        )
    if operation == "preset_locks":
        return preset_locks_to_wire(
            preset_locks(
                package,
                _optional_string(payload.get("preset"), "preset"),
            )
        )
    if operation == "reject_locked_overrides":
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise AdapterProtocolError("Locked-override request requires preset.")
        reject_locked_overrides(
            package,
            preset,
            _optional_object(payload, "overrides", "overrides"),
        )
        return None
    if operation == "validate":
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise AdapterProtocolError("Validation request requires preset.")
        validate_configuration(
            package,
            InspectionRequest(
                preset=preset,
                overrides=_optional_object(payload, "overrides", "overrides"),
                dataset=_optional_string(payload.get("dataset"), "dataset"),
                experiment_task=_optional_string(
                    payload.get("experiment_task"),
                    "experiment_task",
                ),
            ),
        )
        return None
    if operation == "inspect":
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise AdapterProtocolError("Inspection request requires preset.")
        return inspection_result_to_wire(
            inspect_model(
                package,
                InspectionRequest(
                    preset=preset,
                    overrides=_optional_object(payload, "overrides", "overrides"),
                    dataset=_optional_string(payload.get("dataset"), "dataset"),
                    experiment_task=_optional_string(
                        payload.get("experiment_task"),
                        "experiment_task",
                    ),
                ),
            )
        )
    if operation == "parse_search_value":
        search_key = payload.get("search_key")
        if not isinstance(search_key, str):
            raise AdapterProtocolError("Search-value request requires search_key.")
        parsed = parse_config_value(
            package.metadata.search_space,
            search_key,
            str(_scalar(payload.get("value"), "value")),
        )
        if isinstance(parsed, type):
            abstract_error = abstract_config_class_error(parsed)
            if abstract_error is not None:
                raise ValueError(abstract_error)
        return serialize_config_value(parsed)
    if operation == "checkpoint_config_overrides":
        return configuration_values_to_wire(
            package.checkpoint_config_overrides(
                _tensor_shapes(
                    payload.get("tensor_shapes")
                    if payload.get("tensor_shapes") is not None
                    else {}
                )
            )
        )
    if operation == "plan_runs":
        request = run_request_from_wire(_object(payload.get("request"), "request"))
        budget = planning_budget_from_wire(
            _optional_object(payload, "budget", "budget")
        )
        raw_state = payload.get("random_state")
        random_source = None
        if raw_state is not None:
            random_source = random.Random()
            random_source.setstate(random_state_from_wire(raw_state))
        plan = plan_runs(
            package,
            request,
            random_source=random_source,
            budget=budget,
        )
        return {
            "plan": run_plan_to_wire(plan),
            "random_state": (
                random_state_to_wire(random_source.getstate())
                if random_source is not None
                else None
            ),
        }
    if operation == "accept_run_plan":
        request = run_request_from_wire(_object(payload.get("request"), "request"))
        submitted = submitted_runs_from_wire(
            payload.get("runs") if payload.get("runs") is not None else []
        )
        budget = planning_budget_from_wire(
            _optional_object(payload, "budget", "budget")
        )
        return run_plan_to_wire(
            accept_run_plan(package, request, submitted, budget=budget)
        )
    if operation == "execute_run_plan":
        plan = run_plan_from_wire(_object(payload.get("plan"), "plan"))
        progress_path = payload.get("progress_path")
        progress = (
            JsonlRunProgress(Path(_string(progress_path, "progress_path")))
            if progress_path is not None
            else None
        )
        logs_root = _optional_string(payload.get("logs_root"), "logs_root") or "logs"
        log_folder = _optional_string(payload.get("log_folder"), "log_folder")
        progress_step_interval = _positive_integer(
            payload.get("progress_step_interval", 25),
            "progress_step_interval",
        )
        raw_monitors = payload.get("monitors")
        monitors = () if raw_monitors is None else _strings(raw_monitors, "monitors")
        return run_results_to_wire(
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(
                    root=Path(logs_root),
                    namespace=log_folder,
                ),
                progress=progress,
                progress_step_interval=progress_step_interval,
                monitors=monitors,
            )
        )
    raise AdapterProtocolError(f"Unknown Adapter operation: {operation}")


def process_request(request: Mapping[str, Any]) -> dict[str, Any]:
    version = request.get("version")
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise AdapterProtocolError(
            f"Unsupported Adapter protocol version {version!r}; "
            f"expected {PROTOCOL_VERSION}."
        )
    operation = request.get("operation")
    if not isinstance(operation, str):
        raise AdapterProtocolError("Adapter operation must be a string.")
    if "payload" not in request:
        raise AdapterProtocolError("Adapter request requires payload.")
    payload = _object(request["payload"], "payload")
    return {
        "version": PROTOCOL_VERSION,
        "ok": True,
        "result": _handle(operation, payload),
    }


def _response(raw_request: bytes) -> dict[str, Any]:
    if len(raw_request) > MAX_ADAPTER_REQUEST_BYTES:
        return {
            "version": PROTOCOL_VERSION,
            "ok": False,
            "error": {
                "kind": "too-large",
                "type": "AdapterProtocolError",
                "message": "Adapter request exceeds its size limit.",
            },
        }
    try:
        request = json_value_from_wire(json.loads(raw_request))
        return process_request(_object(request, "request"))
    except Exception as exc:  # The process boundary always returns one envelope.
        failure_kind = (
            "invalid"
            if isinstance(
                exc,
                (AdapterProtocolError, InspectionError, RunsError, ValueError),
            )
            else "unavailable"
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        cause = exc.__cause__
        return {
            "version": PROTOCOL_VERSION,
            "ok": False,
            "error": {
                "kind": failure_kind,
                "type": type(exc).__name__,
                "message": str(exc),
                "cause": (
                    {
                        "type": type(cause).__name__,
                        "message": str(cause),
                    }
                    if cause is not None
                    else None
                ),
            },
        }


def _write_response(response: Mapping[str, Any]) -> None:
    json.dump(response, sys.stdout, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    if "--serve" in sys.argv[1:]:
        for raw_request in sys.stdin.buffer:
            _write_response(_response(raw_request))
        return
    raw_request = sys.stdin.buffer.read(MAX_ADAPTER_REQUEST_BYTES + 1)
    _write_response(_response(raw_request))


if __name__ == "__main__":
    main()
