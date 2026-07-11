"""Training Job lifecycle status definitions."""

from __future__ import annotations

LIVE_PROCESS_JOB_STATUSES: frozenset[str] = frozenset({"queued", "running"})
ACTIVE_JOB_STATUSES: frozenset[str] = frozenset(
    {*LIVE_PROCESS_JOB_STATUSES, "unknown"}
)
TERMINAL_JOB_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled"}
)


def is_live_process_job_status(status: str) -> bool:
    return status in LIVE_PROCESS_JOB_STATUSES


def is_active_job_status(status: str) -> bool:
    return status in ACTIVE_JOB_STATUSES


def is_terminal_job_status(status: str) -> bool:
    return status in TERMINAL_JOB_STATUSES
