from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from emperor_workbench.failures import DomainFailure, FailureKind

if TYPE_CHECKING:
    from model_runtime.runs import RunResult


class ProjectAdapterFailure(DomainFailure):
    """The configured model project Adapter rejected or failed a request."""

    def __init__(
        self,
        detail: str,
        *,
        kind: FailureKind = FailureKind.INVALID,
        remote_type: str | None = None,
        remote_cause_detail: str | None = None,
        phase: str | None = None,
        affected_run_id: str | None = None,
        execution_id: str | None = None,
        completed_results: Sequence[RunResult] = (),
    ) -> None:
        super().__init__(detail, kind=kind)
        self.remote_type = remote_type
        self.remote_cause_detail = remote_cause_detail
        self.phase = phase
        self.affected_run_id = affected_run_id
        self.execution_id = execution_id
        self.completed_results = tuple(completed_results)
