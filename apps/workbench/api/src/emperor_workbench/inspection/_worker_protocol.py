from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from model_runtime.packages import model_key

from emperor_workbench.failures import FailureKind
from emperor_workbench.inspection._errors import InspectionFailure

if TYPE_CHECKING:
    from model_runtime.inspection import InspectionRequest, InspectionResult

    from emperor_workbench.inspection._subprocess import InspectionWorkerLimits
    from emperor_workbench.model_packages import SelectedModelPackage

MAX_WORKER_RESULT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class InspectionWorkerRequest:
    model_type: str
    model: str
    preset: str
    overrides: Mapping[str, Any]
    dataset: str | None
    experiment_task: str | None
    memory_bytes: int
    cpu_count: int


def encode_worker_request(
    selected: SelectedModelPackage,
    request: InspectionRequest,
    limits: InspectionWorkerLimits,
) -> bytes:
    from model_runtime.inspection import ParsedOverrides

    raw_overrides = (
        request.overrides.values
        if isinstance(request.overrides, ParsedOverrides)
        else request.overrides
    )
    payload = {
        "modelType": selected.identity.model_type,
        "model": selected.identity.model,
        "preset": request.preset,
        "overrides": selected.serialize_overrides(raw_overrides),
        "dataset": request.dataset,
        "experimentTask": request.experiment_task,
        "limits": {
            "memoryBytes": limits.memory_bytes,
            "cpuCount": limits.cpu_count,
        },
    }
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_worker_request(payload: object) -> InspectionWorkerRequest:
    if not isinstance(payload, Mapping):
        raise ValueError("Inspection worker request must be a mapping.")
    limits = payload.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("Missing Inspection worker limits.")
    memory_bytes = limits.get("memoryBytes")
    cpu_count = limits.get("cpuCount")
    if not isinstance(memory_bytes, int) or memory_bytes < 1:
        raise ValueError("Invalid Inspection worker memory limit.")
    if not isinstance(cpu_count, int) or cpu_count < 1:
        raise ValueError("Invalid Inspection worker CPU limit.")

    model_type = payload.get("modelType")
    model = payload.get("model")
    preset = payload.get("preset")
    overrides = payload.get("overrides")
    dataset = payload.get("dataset")
    experiment_task = payload.get("experimentTask")
    if not isinstance(model_type, str) or not isinstance(model, str):
        raise ValueError("Invalid Inspection worker model identity.")
    try:
        model_key(model_type, model)
    except ValueError as exc:
        raise ValueError("Invalid Inspection worker model identity.") from exc
    if not isinstance(preset, str) or not isinstance(overrides, Mapping):
        raise ValueError("Invalid Inspection worker request.")
    if dataset is not None and not isinstance(dataset, str):
        raise ValueError("Invalid Inspection worker dataset.")
    if experiment_task is not None and not isinstance(experiment_task, str):
        raise ValueError("Invalid Inspection worker experiment task.")
    return InspectionWorkerRequest(
        model_type=model_type,
        model=model,
        preset=preset,
        overrides=overrides,
        dataset=dataset,
        experiment_task=experiment_task,
        memory_bytes=memory_bytes,
        cpu_count=cpu_count,
    )


def success_envelope(result: InspectionResult) -> dict[str, Any]:
    from model_runtime.cli import to_wire

    return {"ok": True, "result": to_wire(result)}


def domain_failure_envelope(exc: InspectionFailure) -> dict[str, Any]:
    return {
        "ok": False,
        "failure": "domain",
        "detail": exc.detail,
        "failureKind": exc.kind.value,
    }


def internal_failure_envelope() -> dict[str, Any]:
    return {
        "ok": False,
        "failure": "internal",
        "detail": "Inspection worker failed.",
        "failureKind": FailureKind.UNAVAILABLE.value,
    }


def decode_worker_response(raw_response: bytes) -> InspectionResult:
    try:
        envelope = json.loads(raw_response)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _invalid_worker_result() from exc
    if not isinstance(envelope, Mapping):
        raise _invalid_worker_result()
    if envelope.get("ok") is False:
        detail = envelope.get("detail")
        raw_kind = envelope.get("failureKind")
        if not isinstance(detail, str) or not isinstance(raw_kind, str):
            raise _invalid_worker_result()
        try:
            failure_kind = FailureKind(raw_kind)
        except ValueError as exc:
            raise _invalid_worker_result() from exc
        raise InspectionFailure(detail, kind=failure_kind)
    if envelope.get("ok") is not True:
        raise _invalid_worker_result()

    try:
        result = envelope["result"]
        if not isinstance(result, Mapping):
            raise TypeError("Inspection result must be a mapping.")
        _validate_result_payload(result)
        from model_runtime.cli import inspection_result_from_wire

        return inspection_result_from_wire(result)
    except (KeyError, TypeError, ValueError) as exc:
        raise _invalid_worker_result() from exc


def _validate_result_payload(payload: Mapping[str, Any]) -> None:
    identity = _mapping(payload["identity"])
    _string(identity["model_type"])
    _string(identity["model"])
    _string(payload["preset"])
    _integer(payload["parameter_count"])
    _integer(payload["parameter_size_bytes"])

    nodes = _list(payload["nodes"])
    edges = _list(payload["edges"])
    for node_value in nodes:
        node = _mapping(node_value)
        _string(node["id"])
        _string(node["type_name"])
        _optional_string(node["description"])
        _string(node["path"])
        if node["graph_role"] not in {"architecture", "internal", "runtime"}:
            raise ValueError("Invalid graph role.")
        _integer(node["parameter_count"])
        _integer(node["parameter_size_bytes"])
        _mapping(node["details"])
        configuration = node["configuration"]
        if configuration is not None:
            config = _mapping(configuration)
            _string(config["type_name"])
            for field_value in _list(config["fields"]):
                field = _mapping(field_value)
                _string(field["key"])
                _optional_string(field["description"])
                field["value"]
    for edge_value in edges:
        edge = _mapping(edge_value)
        _string(edge["id"])
        _string(edge["source"])
        _string(edge["target"])


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected mapping value.")
    return value


def _list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("Expected list value.")
    return value


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected string value.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("Expected integer value.")
    return value


def _invalid_worker_result() -> InspectionFailure:
    return InspectionFailure(
        "Inspection worker produced an invalid result.",
        kind=FailureKind.UNAVAILABLE,
    )


__all__: list[str] = []
