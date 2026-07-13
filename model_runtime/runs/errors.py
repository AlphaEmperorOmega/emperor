from __future__ import annotations


class RunsError(Exception):
    """A selected Model Package cannot satisfy a Runs operation."""


class InvalidRunRequest(RunsError):
    """A Run request is invalid for the selected Model Package."""


class InvalidRunPlan(RunsError):
    """A submitted or reloaded Run Plan is invalid."""


class InvalidCheckpointContinuation(RunsError):
    """A checkpoint cannot continue the selected Run."""


class PlanTooLarge(InvalidRunRequest):
    """A caller-supplied planning budget would be exceeded."""


__all__ = [
    "InvalidCheckpointContinuation",
    "InvalidRunPlan",
    "InvalidRunRequest",
    "PlanTooLarge",
    "RunsError",
]
