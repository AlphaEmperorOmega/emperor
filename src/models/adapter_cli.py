from __future__ import annotations

import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model_runtime.cli import (
    PROTOCOL_VERSION,
    package_metadata_to_wire,
    planning_budget_from_wire,
    run_plan_from_wire,
    run_request_from_wire,
    submitted_run_from_wire,
    to_wire,
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
    JsonlTrainingProgressCallback,
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


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
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
    task = package.resolve_experiment_task(payload.get("experiment_task"))
    raw_presets = payload.get("presets") or []
    raw_datasets = payload.get("datasets") or []
    raw_monitors = payload.get("monitors") or []
    if not isinstance(raw_presets, list) or not isinstance(raw_datasets, list):
        raise AdapterProtocolError("Resolve presets and datasets must be lists.")
    if not isinstance(raw_monitors, list):
        raise AdapterProtocolError("Resolve monitors must be a list.")
    presets = [package.resolve_preset(str(name)) for name in raw_presets]
    datasets = package.resolve_datasets([str(name) for name in raw_datasets], task)
    monitors = package.resolve_monitors([str(name) for name in raw_monitors])
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
        return to_wire(configuration_schema(package, payload.get("preset")))
    if operation == "search_space":
        raw_presets = payload.get("presets")
        presets = (
            tuple(str(item) for item in raw_presets)
            if isinstance(raw_presets, list)
            else None
        )
        return to_wire(
            search_space_schema(
                package,
                payload.get("preset"),
                presets,
            )
        )
    if operation in {"parse_overrides", "serialize_overrides"}:
        overrides = _object(payload.get("overrides") or {}, "overrides")
        if operation == "parse_overrides":
            return to_wire(
                parse_overrides(
                    package,
                    overrides,
                    preset=payload.get("preset"),
                    ignore_unknown=bool(payload.get("ignore_unknown", False)),
                ).values
            )
        return to_wire(
            serialize_overrides(
                package,
                overrides,
                ignore_unknown=bool(payload.get("ignore_unknown", False)),
            )
        )
    if operation == "preset_locks":
        return to_wire(preset_locks(package, payload.get("preset")))
    if operation == "reject_locked_overrides":
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise AdapterProtocolError("Locked-override request requires preset.")
        reject_locked_overrides(
            package,
            preset,
            _object(payload.get("overrides") or {}, "overrides"),
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
                overrides=_object(payload.get("overrides") or {}, "overrides"),
                dataset=payload.get("dataset"),
                experiment_task=payload.get("experiment_task"),
            ),
        )
        return None
    if operation == "inspect":
        preset = payload.get("preset")
        if not isinstance(preset, str):
            raise AdapterProtocolError("Inspection request requires preset.")
        return to_wire(
            inspect_model(
                package,
                InspectionRequest(
                    preset=preset,
                    overrides=_object(payload.get("overrides") or {}, "overrides"),
                    dataset=payload.get("dataset"),
                    experiment_task=payload.get("experiment_task"),
                ),
            )
        )
    if operation == "parse_search_value":
        search_key = payload.get("search_key")
        if not isinstance(search_key, str):
            raise AdapterProtocolError("Search-value request requires search_key.")
        parsed = parse_config_value(
            package.metadata.search_space_module,
            search_key,
            str(payload.get("value")),
        )
        if isinstance(parsed, type):
            abstract_error = abstract_config_class_error(parsed)
            if abstract_error is not None:
                raise ValueError(abstract_error)
        return serialize_config_value(parsed)
    if operation == "checkpoint_config_overrides":
        return to_wire(
            package.checkpoint_config_overrides(
                _tensor_shapes(payload.get("tensor_shapes") or {})
            )
        )
    if operation == "plan_runs":
        request = run_request_from_wire(_object(payload.get("request"), "request"))
        budget = planning_budget_from_wire(
            _object(payload.get("budget") or {}, "budget")
        )
        raw_state = payload.get("random_state")
        random_source = None
        if raw_state is not None:
            random_source = random.Random()
            random_source.setstate(_tuple_tree(raw_state))
        plan = plan_runs(
            package,
            request,
            random_source=random_source,
            budget=budget,
        )
        return {
            "plan": to_wire(plan),
            "random_state": (
                to_wire(random_source.getstate())
                if random_source is not None
                else None
            ),
        }
    if operation == "accept_run_plan":
        request = run_request_from_wire(_object(payload.get("request"), "request"))
        submitted = tuple(
            submitted_run_from_wire(_object(item, "submitted run"))
            for item in payload.get("runs") or ()
        )
        budget = planning_budget_from_wire(
            _object(payload.get("budget") or {}, "budget")
        )
        return to_wire(accept_run_plan(package, request, submitted, budget=budget))
    if operation == "execute_run_plan":
        plan = run_plan_from_wire(_object(payload.get("plan"), "plan"))
        progress_path = payload.get("progress_path")
        progress = (
            JsonlTrainingProgressCallback(
                Path(str(progress_path)),
                step_interval=int(payload.get("progress_step_interval") or 25),
            )
            if progress_path is not None
            else None
        )
        return to_wire(
            execute_runs(
                package,
                plan,
                artifacts=FilesystemRunArtifacts(
                    root=Path(str(payload.get("logs_root") or "logs")),
                    namespace=(
                        str(payload["log_folder"])
                        if payload.get("log_folder")
                        else None
                    ),
                ),
                progress=progress,
                monitors=tuple(str(item) for item in payload.get("monitors") or ()),
            )
        )
    raise AdapterProtocolError(f"Unknown Adapter operation: {operation}")


def process_request(request: Mapping[str, Any]) -> dict[str, Any]:
    version = request.get("version")
    if version != PROTOCOL_VERSION:
        raise AdapterProtocolError(
            f"Unsupported Adapter protocol version {version!r}; "
            f"expected {PROTOCOL_VERSION}."
        )
    operation = request.get("operation")
    if not isinstance(operation, str):
        raise AdapterProtocolError("Adapter operation must be a string.")
    payload = _object(request.get("payload") or {}, "payload")
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
        request = json.loads(raw_request)
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
