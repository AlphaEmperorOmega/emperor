from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from typing import Any

from model_runtime.cli import PROTOCOL_VERSION, run_result_from_wire
from model_runtime.runs import RunResult

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter._errors import ProjectAdapterFailure

PROJECT_ADAPTER_COMMAND_ENV = "EMPEROR_PROJECT_ADAPTER_COMMAND"
MAX_PROJECT_ADAPTER_REQUEST_BYTES = 64 * 1024 * 1024
MAX_PROJECT_ADAPTER_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_PARTIAL_RUN_RESULTS = 2_000
_RUN_EXECUTION_PHASES = {
    "training",
    "result_commit",
    "best_results_projection",
    "progress_projection",
}


class ProjectAdapterProtocolFailure(ProjectAdapterFailure):
    """The Adapter process returned data that violates the wire protocol."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, kind=FailureKind.UNAVAILABLE)


def default_project_adapter_command() -> tuple[str, ...]:
    configured = os.environ.get(PROJECT_ADAPTER_COMMAND_ENV)
    if configured:
        command = tuple(shlex.split(configured))
        if command:
            return command
    installed = shutil.which("emperor-project-adapter")
    if installed:
        return (installed,)
    return (sys.executable, "-m", "models.adapter_cli")


def encode_request(
    operation: str,
    payload: dict[str, Any] | None,
    *,
    line_delimited: bool,
) -> bytes:
    encoded = json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "operation": operation,
            "payload": payload or {},
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    wire_size = len(encoded) + int(line_delimited)
    if wire_size > MAX_PROJECT_ADAPTER_REQUEST_BYTES:
        raise ProjectAdapterFailure(
            "The model project Adapter request exceeded its size limit.",
            kind=FailureKind.TOO_LARGE,
        )
    return encoded


def decode_response(raw_response: bytes) -> Any:
    try:
        envelope = json.loads(
            raw_response,
            parse_constant=_reject_nonfinite_json,
        )
    except (ValueError, RecursionError) as exc:
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an invalid response."
        ) from exc
    if (
        not isinstance(envelope, dict)
        or type(envelope.get("version")) is not int
        or envelope["version"] != PROTOCOL_VERSION
    ):
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an incompatible response."
        )
    if envelope.get("ok") is not True:
        error = envelope.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            raise ProjectAdapterProtocolFailure(
                "The model project Adapter returned an invalid failure."
            )
        try:
            kind = FailureKind(str(error.get("kind")))
        except ValueError:
            kind = FailureKind.UNAVAILABLE
        cause = error.get("cause")
        partial = _partial_plan_failure(error)
        raise ProjectAdapterFailure(
            error["message"],
            kind=kind,
            remote_type=(str(error["type"]) if error.get("type") is not None else None),
            remote_cause_detail=(
                str(cause["message"])
                if isinstance(cause, dict) and cause.get("message") is not None
                else None
            ),
            phase=partial[0],
            affected_run_id=partial[1],
            execution_id=partial[2],
            completed_results=partial[3],
        )
    return envelope.get("result")


def _partial_plan_failure(
    error: dict[str, Any],
) -> tuple[str | None, str | None, str | None, tuple[RunResult, ...]]:
    fields = ("phase", "affected_run_id", "execution_id", "completed_results")
    present = [field in error for field in fields]
    if not any(present):
        return None, None, None, ()
    if not all(present):
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an invalid partial Run failure."
        )
    phase = error["phase"]
    affected_run_id = error["affected_run_id"]
    execution_id = error["execution_id"]
    raw_results = error["completed_results"]
    if (
        type(phase) is not str
        or phase not in _RUN_EXECUTION_PHASES
        or type(affected_run_id) is not str
        or not affected_run_id
        or type(execution_id) is not str
        or not execution_id
        or len(execution_id) > 128
        or not isinstance(raw_results, list)
        or not raw_results
        or len(raw_results) > MAX_PARTIAL_RUN_RESULTS
    ):
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an invalid partial Run failure."
        )
    try:
        results = tuple(run_result_from_wire(result) for result in raw_results)
    except (TypeError, ValueError) as exc:
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an invalid partial Run failure."
        ) from exc
    completed_run_ids = {result.run_id for result in results}
    affected_is_completed = affected_run_id in completed_run_ids
    post_commit = phase in {"best_results_projection", "progress_projection"}
    if affected_is_completed != post_commit:
        raise ProjectAdapterProtocolFailure(
            "The model project Adapter returned an invalid partial Run failure."
        )
    return phase, affected_run_id, execution_id, results


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {value}")


def require_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectAdapterProtocolFailure("The project Adapter result is invalid.")
    return value


def require_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ProjectAdapterProtocolFailure("The project Adapter result is invalid.")
    return value


def require_string(value: object) -> str:
    if not isinstance(value, str):
        raise ProjectAdapterProtocolFailure("The project Adapter result is invalid.")
    return value


def require_field(mapping: dict[str, Any], name: str) -> Any:
    try:
        return mapping[name]
    except KeyError as exc:
        raise ProjectAdapterProtocolFailure(
            "The project Adapter result is invalid."
        ) from exc
