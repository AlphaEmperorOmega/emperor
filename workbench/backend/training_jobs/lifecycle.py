from __future__ import annotations

from typing import Any

from workbench.backend.training_jobs.status import is_terminal_job_status

_TERMINAL_EVENT_TYPE_STATUS: dict[str, str] = {
    "completed": "completed",
    "failed": "failed",
    "error": "failed",
    "cancelled": "cancelled",
}


def terminal_status_from_event(event: dict[str, Any]) -> str | None:
    status = event.get("status")
    if isinstance(status, str) and is_terminal_job_status(status):
        return status
    event_type = event.get("type")
    if isinstance(event_type, str):
        return _TERMINAL_EVENT_TYPE_STATUS.get(event_type)
    return None


def latest_terminal_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in reversed(events)
            if terminal_status_from_event(event) is not None
        ),
        None,
    )


def terminal_exit_code(
    status: str,
    event: dict[str, Any],
    current_exit_code: int | None,
) -> int | None:
    if event.get("type") == "operator_reconciled_failed":
        return None
    explicit_exit_code = event.get("exitCode")
    if isinstance(explicit_exit_code, int):
        return explicit_exit_code
    if current_exit_code is not None:
        return current_exit_code
    if status == "completed":
        return 0
    if status == "failed":
        return 1
    if status == "cancelled":
        return -15
    return None


__all__ = [
    "latest_terminal_event",
    "terminal_exit_code",
    "terminal_status_from_event",
]
