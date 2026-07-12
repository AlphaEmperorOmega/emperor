"""Killable process boundary for model construction during Inspection."""

from __future__ import annotations

import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from workbench.backend.failures import FailureKind
from workbench.backend.inspection_errors import InspectionFailure

if TYPE_CHECKING:
    from emperor.inspection import InspectionRequest, InspectionResult
    from emperor.model_packages import ModelPackage

_DEFAULT_WORKER_COMMAND = (
    sys.executable,
    "-P",
    "-m",
    "workbench.backend.inspection_worker",
)
_MAX_WORKER_RESULT_BYTES = 16 * 1024 * 1024
_WORKER_IMPORT_ROOT = str(Path(__file__).resolve().parents[2])


@dataclass(frozen=True, slots=True)
class InspectionWorkerLimits:
    memory_bytes: int = 4 * 1024**3
    cpu_count: int = 4
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.memory_bytes < 1:
            raise ValueError("Inspection memory limit must be positive.")
        if self.cpu_count < 1:
            raise ValueError("Inspection CPU limit must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("Inspection timeout must be positive.")


class InspectionExecutor(Protocol):
    def inspect(
        self,
        package: ModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult: ...


class InProcessInspectionExecutor:
    """Semantic executor retained for CLI, compatibility, and focused tests."""

    def inspect(
        self,
        package: ModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult:
        from workbench.backend.inspection_adapter import (
            WorkbenchInspectionAdapter,
        )

        return WorkbenchInspectionAdapter.from_package(package).inspect(request)


class SubprocessInspectionExecutor:
    """Execute one semantic Inspection request in a fresh process group."""

    def __init__(
        self,
        limits: InspectionWorkerLimits | None = None,
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        self._limits = limits or InspectionWorkerLimits()
        self._command = tuple(command or _DEFAULT_WORKER_COMMAND)

    def inspect(
        self,
        package: ModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult:
        from emperor.inspection import ParsedOverrides

        from workbench.backend.inspection_adapter import (
            WorkbenchInspectionAdapter,
        )

        adapter = WorkbenchInspectionAdapter.from_package(package)
        raw_overrides = (
            request.overrides.values
            if isinstance(request.overrides, ParsedOverrides)
            else request.overrides
        )
        payload = {
            "modelType": package.identity.model_type,
            "model": package.identity.model,
            "preset": request.preset,
            "overrides": adapter.serialize_overrides(raw_overrides),
            "dataset": request.dataset,
            "experimentTask": request.experiment_task,
            "limits": {
                "memoryBytes": self._limits.memory_bytes,
                "cpuCount": self._limits.cpu_count,
            },
        }
        encoded_request = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        process = subprocess.Popen(  # noqa: S603 - fixed or injected test command
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": _WORKER_IMPORT_ROOT},
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate(
                encoded_request,
                timeout=self._limits.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            process.communicate()
            raise InspectionFailure(
                "Inspection construction exceeded the "
                f"{self._limits.timeout_seconds:g} second limit.",
                kind=FailureKind.TIMEOUT,
            ) from exc

        if process.returncode != 0:
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker crashed.",
                kind=FailureKind.UNAVAILABLE,
            )
        if len(stdout) > _MAX_WORKER_RESULT_BYTES:
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker result exceeded its size limit.",
                kind=FailureKind.UNAVAILABLE,
            )
        try:
            envelope = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker produced an invalid result.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        if not isinstance(envelope, dict):
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker produced an invalid result.",
                kind=FailureKind.UNAVAILABLE,
            )
        if envelope.get("ok") is False:
            detail = envelope.get("detail")
            raw_kind = envelope.get("failureKind")
            if not isinstance(detail, str) or not isinstance(raw_kind, str):
                _terminate_process_group(process)
                raise InspectionFailure(
                    "Inspection worker produced an invalid result.",
                    kind=FailureKind.UNAVAILABLE,
                )
            try:
                failure_kind = FailureKind(raw_kind)
            except ValueError as exc:
                _terminate_process_group(process)
                raise InspectionFailure(
                    "Inspection worker produced an invalid result.",
                    kind=FailureKind.UNAVAILABLE,
                ) from exc
            if envelope.get("failure") != "domain":
                _terminate_process_group(process)
            raise InspectionFailure(detail, kind=failure_kind)
        if envelope.get("ok") is not True:
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker produced an invalid result.",
                kind=FailureKind.UNAVAILABLE,
            )
        try:
            return _inspection_result_from_wire(envelope["result"])
        except (KeyError, TypeError, ValueError) as exc:
            _terminate_process_group(process)
            raise InspectionFailure(
                "Inspection worker produced an invalid result.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _apply_worker_limits(payload: Mapping[str, Any]) -> None:
    limits = payload.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("Missing Inspection worker limits.")
    memory_bytes = limits.get("memoryBytes")
    cpu_count = limits.get("cpuCount")
    if not isinstance(memory_bytes, int) or memory_bytes < 1:
        raise ValueError("Invalid Inspection worker memory limit.")
    if not isinstance(cpu_count, int) or cpu_count < 1:
        raise ValueError("Invalid Inspection worker CPU limit.")
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = sorted(os.sched_getaffinity(0))
        selected_cpus = available_cpus[: min(cpu_count, len(available_cpus))]
        if selected_cpus:
            os.sched_setaffinity(0, selected_cpus)


def _run_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    _apply_worker_limits(payload)
    from emperor.inspection import InspectionRequest

    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    model_type = payload.get("modelType")
    model = payload.get("model")
    preset = payload.get("preset")
    overrides = payload.get("overrides")
    dataset = payload.get("dataset")
    experiment_task = payload.get("experimentTask")
    if not isinstance(model_type, str) or not isinstance(model, str):
        raise ValueError("Invalid Inspection worker model identity.")
    if not isinstance(preset, str) or not isinstance(overrides, Mapping):
        raise ValueError("Invalid Inspection worker request.")
    if dataset is not None and not isinstance(dataset, str):
        raise ValueError("Invalid Inspection worker dataset.")
    if experiment_task is not None and not isinstance(experiment_task, str):
        raise ValueError("Invalid Inspection worker experiment task.")
    adapter = WorkbenchInspectionAdapter.select_parts(model_type, model)
    parsed_overrides = adapter.parse_overrides(overrides)
    result = adapter.inspect(
        InspectionRequest(
            preset=preset,
            overrides=parsed_overrides,
            dataset=dataset,
            experiment_task=experiment_task,
        )
    )
    return {"ok": True, "result": _inspection_result_to_wire(result)}


def _inspection_result_to_wire(result: InspectionResult) -> dict[str, Any]:
    return {
        "identity": {
            "model_type": result.identity.model_type,
            "model": result.identity.model,
        },
        "preset": result.preset,
        "parameter_count": result.parameter_count,
        "parameter_size_bytes": result.parameter_size_bytes,
        "nodes": [
            {
                "id": node.id,
                "type_name": node.type_name,
                "description": node.description,
                "path": node.path,
                "graph_role": node.graph_role,
                "parameter_count": node.parameter_count,
                "parameter_size_bytes": node.parameter_size_bytes,
                "details": _thaw(node.details),
                "configuration": (
                    None
                    if node.configuration is None
                    else {
                        "type_name": node.configuration.type_name,
                        "fields": [
                            {
                                "key": field.key,
                                "value": _thaw(field.value),
                                "description": field.description,
                            }
                            for field in node.configuration.fields
                        ],
                    }
                ),
            }
            for node in result.nodes
        ],
        "edges": [
            {"id": edge.id, "source": edge.source, "target": edge.target}
            for edge in result.edges
        ],
    }


def _inspection_result_from_wire(payload: object) -> InspectionResult:
    from emperor.inspection import GraphEdge, InspectionResult
    from emperor.model_packages import ModelIdentity

    if not isinstance(payload, Mapping):
        raise TypeError("Inspection result must be a mapping.")
    identity_payload = payload["identity"]
    if not isinstance(identity_payload, Mapping):
        raise TypeError("Inspection identity must be a mapping.")
    nodes_payload = payload["nodes"]
    edges_payload = payload["edges"]
    if not isinstance(nodes_payload, list) or not isinstance(edges_payload, list):
        raise TypeError("Inspection graph must contain node and edge lists.")
    return InspectionResult(
        identity=ModelIdentity(
            model_type=_string(identity_payload["model_type"]),
            model=_string(identity_payload["model"]),
        ),
        preset=_string(payload["preset"]),
        parameter_count=_integer(payload["parameter_count"]),
        parameter_size_bytes=_integer(payload["parameter_size_bytes"]),
        nodes=tuple(_node_from_wire(node) for node in nodes_payload),
        edges=tuple(
            GraphEdge(
                id=_string(edge["id"]),
                source=_string(edge["source"]),
                target=_string(edge["target"]),
            )
            for edge in edges_payload
            if isinstance(edge, Mapping)
        ),
    )


def _node_from_wire(payload: object):
    from emperor.inspection import (
        GraphConfiguration,
        GraphConfigurationField,
        GraphNode,
    )

    if not isinstance(payload, Mapping):
        raise TypeError("Inspection node must be a mapping.")
    configuration_payload = payload["configuration"]
    configuration = None
    if configuration_payload is not None:
        if not isinstance(configuration_payload, Mapping):
            raise TypeError("Inspection configuration must be a mapping.")
        fields_payload = configuration_payload["fields"]
        if not isinstance(fields_payload, list):
            raise TypeError("Inspection configuration fields must be a list.")
        configuration = GraphConfiguration(
            type_name=_string(configuration_payload["type_name"]),
            fields=tuple(
                GraphConfigurationField(
                    key=_string(field["key"]),
                    value=field["value"],
                    description=_optional_string(field["description"]),
                )
                for field in fields_payload
                if isinstance(field, Mapping)
            ),
        )
    details = payload["details"]
    if not isinstance(details, Mapping):
        raise TypeError("Inspection node details must be a mapping.")
    return GraphNode(
        id=_string(payload["id"]),
        type_name=_string(payload["type_name"]),
        description=_optional_string(payload["description"]),
        path=_string(payload["path"]),
        graph_role=_graph_role(payload["graph_role"]),
        parameter_count=_integer(payload["parameter_count"]),
        parameter_size_bytes=_integer(payload["parameter_size_bytes"]),
        details=details,
        configuration=configuration,
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
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


def _graph_role(value: object):
    if value not in {"architecture", "internal", "runtime"}:
        raise ValueError("Invalid graph role.")
    return value


def main() -> int:
    protocol_stdout = sys.stdout
    try:
        payload = json.loads(sys.stdin.buffer.read())
        if not isinstance(payload, Mapping):
            raise ValueError("Inspection worker request must be a mapping.")
        with contextlib.redirect_stdout(sys.stderr):
            envelope = _run_worker(payload)
    except InspectionFailure as exc:
        envelope = {
            "ok": False,
            "failure": "domain",
            "detail": exc.detail,
            "failureKind": exc.kind.value,
        }
    except Exception:
        envelope = {
            "ok": False,
            "failure": "internal",
            "detail": "Inspection worker failed.",
            "failureKind": FailureKind.UNAVAILABLE.value,
        }
    try:
        json.dump(
            envelope,
            protocol_stdout,
            allow_nan=False,
            separators=(",", ":"),
        )
    except Exception:
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InProcessInspectionExecutor",
    "InspectionExecutor",
    "InspectionWorkerLimits",
    "SubprocessInspectionExecutor",
]
