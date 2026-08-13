from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast, get_args

from model_runtime.runs.records import RunResult

RunExecutionPhase = Literal[
    "training",
    "result_commit",
    "best_results_projection",
    "progress_projection",
]
_RUN_EXECUTION_PHASES = frozenset(cast(tuple[str, ...], get_args(RunExecutionPhase)))
_POST_COMMIT_PHASES = {"best_results_projection", "progress_projection"}


class RunsError(Exception):
    """A selected Model Package cannot satisfy a Runs operation."""


class InvalidRunRequest(RunsError):
    """A Run request is invalid for the selected Model Package."""


class InvalidRunPlan(RunsError):
    """A submitted or reloaded Run Plan is invalid."""


class InvalidCheckpointContinuation(RunsError):
    """A checkpoint cannot continue the selected Run."""


class RunPlanExecutionError(RunsError):
    """A Run Plan stopped after at least one Run receipt was committed.

    Pre-commit phases exclude the affected Run from ``completed_results``;
    post-commit projection phases include it.
    """

    def __init__(
        self,
        *,
        completed_results: Sequence[RunResult],
        affected_run_id: str,
        phase: RunExecutionPhase,
        execution_id: str,
    ) -> None:
        untrusted_results = tuple(cast(Sequence[object], completed_results))
        if not untrusted_results:
            raise ValueError("completed_results must contain at least one RunResult.")
        if any(not isinstance(result, RunResult) for result in untrusted_results):
            raise TypeError("completed_results must contain only RunResult values.")
        if type(affected_run_id) is not str or not affected_run_id:
            raise ValueError("affected_run_id must be a non-empty string.")
        if type(execution_id) is not str or not execution_id or len(execution_id) > 128:
            raise ValueError("execution_id must be a non-empty bounded string.")
        untrusted_phase = cast(object, phase)
        if (
            type(untrusted_phase) is not str
            or untrusted_phase not in _RUN_EXECUTION_PHASES
        ):
            raise ValueError("phase must be a recognized Run execution phase.")
        completed = cast(tuple[RunResult, ...], untrusted_results)
        affected_is_completed = any(
            result.run_id == affected_run_id for result in completed
        )
        if affected_is_completed != (phase in _POST_COMMIT_PHASES):
            raise ValueError(
                "completed_results do not match the affected Run execution phase."
            )
        super().__init__(
            f"Run Plan execution stopped during {phase} for Run "
            f"'{affected_run_id}' after {len(completed)} completed Run(s)."
        )
        self.completed_results = completed
        self.affected_run_id = affected_run_id
        self.phase = phase
        self.execution_id = execution_id


class PlanTooLarge(InvalidRunRequest):
    """A caller-supplied planning budget would be exceeded."""


__all__ = [
    "InvalidCheckpointContinuation",
    "InvalidRunPlan",
    "InvalidRunRequest",
    "PlanTooLarge",
    "RunPlanExecutionError",
    "RunsError",
]
