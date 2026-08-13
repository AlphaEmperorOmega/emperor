from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from emperor.experiments import ExperimentTask
from model_runtime.packages import ModelPackage
from model_runtime.runs._handoff import TrainingRunRequest
from model_runtime.runs._value_policy import deep_freeze, deep_thaw
from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts
from model_runtime.runs.errors import (
    InvalidRunPlan,
    InvalidRunRequest,
    RunExecutionPhase,
    RunPlanExecutionError,
)
from model_runtime.runs.records import RunPlan, RunPlanRetry, RunResult, RunSpec


def _empty_artifact_ids() -> set[str]:
    return set()


def new_execution_id() -> str:
    """Return an opaque identity for one fresh Run Plan execution."""

    return uuid4().hex


def selected_retry(value: object) -> RunPlanRetry | None:
    if value is not None and not isinstance(value, RunPlanRetry):
        raise TypeError("Run Plan retry must be a RunPlanRetry or None.")
    return value


def retry_prefix(
    plan: RunPlan,
    retry: RunPlanRetry,
) -> tuple[tuple[RunSpec, RunResult], ...]:
    completed = tuple(retry.completed_results)
    if len(completed) > len(plan.runs):
        raise ValueError("Run Plan retry contains more results than the selected Plan.")
    pairs = tuple(zip(plan.runs[: len(completed)], completed, strict=True))
    for position, (run, result) in enumerate(pairs, start=1):
        expected = (run.id, run.experiment_task, run.preset, run.dataset)
        observed = (
            result.run_id,
            result.experiment_task,
            result.preset,
            result.dataset,
        )
        if observed != expected:
            raise ValueError(
                f"Run Plan retry result at position {position} does not match "
                "the selected Run prefix."
            )
    return pairs


@dataclass(slots=True)
class _RetryReceiptValidator:
    identity_payload: Mapping[str, str]
    retry: RunPlanRetry
    seen_artifact_ids: set[str] = field(default_factory=_empty_artifact_ids)

    def validate(
        self,
        run: RunSpec,
        request: TrainingRunRequest,
        result: RunResult,
        receipt: Mapping[str, Any],
    ) -> None:
        if receipt != _json_mapping(result.payload):
            raise ValueError(
                "The persisted result does not match its supplied RunResult."
            )
        artifact_id = receipt.get("artifactId")
        if type(artifact_id) is not str or not artifact_id:
            raise ValueError("The persisted result has no artifact attempt identity.")
        if artifact_id in self.seen_artifact_ids:
            raise ValueError("The Run Plan retry repeats an artifact attempt identity.")
        self.seen_artifact_ids.add(artifact_id)
        expected: dict[str, object] = {
            **self.identity_payload,
            "status": "completed",
            "artifactId": artifact_id,
            "executionId": self.retry.execution_id,
            "runId": run.id,
            "experimentTask": run.experiment_task,
            "dataset": run.dataset,
            "preset": run.preset,
            "presetKey": getattr(request.preset, "name", None),
            "params": _json_mapping(request.parameters),
        }
        mismatched = [
            key for key, value in expected.items() if receipt.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                "The persisted result does not match its selected Run semantics: "
                + ", ".join(mismatched)
                + "."
            )


@dataclass(frozen=True, slots=True)
class RunRetryContext:
    package: ModelPackage
    plan: RunPlan
    requests: Sequence[TrainingRunRequest]
    experiment_task: ExperimentTask | None
    artifacts: RunArtifacts


def admit_run_plan_retry(
    context: RunRetryContext,
    retry: RunPlanRetry | None,
) -> tuple[RunResult, ...]:
    if retry is None:
        return ()
    if not isinstance(context.artifacts, FilesystemRunArtifacts):
        raise InvalidRunRequest("Run Plan retry requires FilesystemRunArtifacts.")
    try:
        pairs = retry_prefix(context.plan, retry)
        receipts = _admitted_retry_receipts(
            context.package,
            context.requests,
            context.artifacts,
            retry,
            pairs,
        )
    except (OSError, ValueError) as exc:
        raise InvalidRunPlan(f"Run Plan retry receipt is invalid: {exc}") from exc
    for (run, _result), receipt in zip(pairs, receipts, strict=True):
        try:
            context.artifacts.update_best_results(
                context.package.identity,
                context.experiment_task,
                receipt,
            )
        except Exception as exc:
            raise RunPlanExecutionError(
                completed_results=retry.completed_results,
                affected_run_id=run.id,
                phase="best_results_projection",
                execution_id=retry.execution_id,
            ) from exc
    return tuple(retry.completed_results)


def _admitted_retry_receipts(
    package: ModelPackage,
    requests: Sequence[TrainingRunRequest],
    artifacts: FilesystemRunArtifacts,
    retry: RunPlanRetry,
    pairs: Sequence[tuple[RunSpec, RunResult]],
) -> tuple[dict[str, Any], ...]:
    receipts: list[dict[str, Any]] = []
    validator = _RetryReceiptValidator(package.identity.to_payload(), retry)
    for (run, result), request in zip(
        pairs,
        requests[: len(pairs)],
        strict=True,
    ):
        if type(result.log_dir) is not str:
            raise ValueError("The supplied RunResult has an invalid artifact path.")
        receipt = artifacts.read_result(result.log_dir)
        validator.validate(run, request, result, receipt)
        receipts.append(receipt)
    return tuple(receipts)


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = deep_thaw(value)
    encoded = json.dumps(projected, allow_nan=False, default=str, sort_keys=True)
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("A Run result must project to a JSON object.")
    return cast(dict[str, Any], decoded)


def _new_artifact_id() -> str:
    return uuid4().hex


def run_result(
    run: RunSpec,
    payload: Mapping[str, Any],
    log_dir: str,
) -> RunResult:
    return RunResult(
        run_id=run.id,
        experiment_task=run.experiment_task,
        preset=run.preset,
        dataset=run.dataset,
        log_dir=log_dir,
        payload=payload,
    )


@dataclass(slots=True)
class TrainingOutcomeObserver:
    """Internal state shared by one Experiment and its Runs executor."""

    execution_id: str | None = None
    artifact_id: str = field(default_factory=_new_artifact_id)
    phase: RunExecutionPhase = "training"
    committed_payload: Mapping[str, Any] | None = None
    committed_log_dir: str | None = None

    @property
    def is_committed(self) -> bool:
        return self.committed_payload is not None

    def receipt_fields(
        self,
        run_id: str | None,
    ) -> dict[str, str]:
        fields = {
            "status": "completed",
            "artifactId": self.artifact_id,
        }
        if run_id is not None:
            fields["runId"] = run_id
        if self.execution_id is not None:
            fields["executionId"] = self.execution_id
        return fields

    def begin_result_commit(self) -> None:
        self.phase = "result_commit"

    @staticmethod
    def prepare_commit(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], deep_freeze(payload))

    def commit(self, payload: Mapping[str, Any], log_dir: str) -> None:
        self.committed_payload = payload
        self.committed_log_dir = log_dir

    def project(
        self,
        best_results: Callable[[], object],
        progress: Callable[[], object],
    ) -> None:
        failures: list[tuple[RunExecutionPhase, Exception]] = []
        projections: tuple[
            tuple[RunExecutionPhase, Callable[[], object]],
            ...,
        ] = (
            ("best_results_projection", best_results),
            ("progress_projection", progress),
        )
        for phase, operation in projections:
            self.phase = phase
            try:
                operation()
            except Exception as exc:
                failures.append((phase, exc))
        if not failures:
            return
        failed_phase, primary = failures[0]
        self.phase = failed_phase
        for secondary_phase, secondary in failures[1:]:
            primary.add_note(
                f"A second {secondary_phase} failure occurred: "
                f"{type(secondary).__name__}: {secondary}"
            )
        raise primary


__all__: list[str] = []
