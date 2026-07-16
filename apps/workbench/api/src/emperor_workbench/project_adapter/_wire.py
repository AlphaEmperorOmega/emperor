from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from typing import Any

from model_runtime.cli import PROTOCOL_VERSION

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter._errors import ProjectAdapterFailure

PROJECT_ADAPTER_COMMAND_ENV = "EMPEROR_PROJECT_ADAPTER_COMMAND"
MAX_PROJECT_ADAPTER_REQUEST_BYTES = 64 * 1024 * 1024
MAX_PROJECT_ADAPTER_RESPONSE_BYTES = 64 * 1024 * 1024


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
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
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
        raise ProjectAdapterFailure(
            error["message"],
            kind=kind,
            remote_type=(str(error["type"]) if error.get("type") is not None else None),
            remote_cause_detail=(
                str(cause["message"])
                if isinstance(cause, dict) and cause.get("message") is not None
                else None
            ),
        )
    return envelope.get("result")


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


def tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tuple_tree(item) for item in value)
    return value
