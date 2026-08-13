from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_runtime.runs.artifacts import FilesystemRunArtifacts, RunArtifacts
    from model_runtime.runs.checkpoint_admission import (
        DEFAULT_CHECKPOINT_ADMISSION_POLICY,
        CheckpointAdmissionPolicy,
    )
    from model_runtime.runs.checkpoints import CheckpointContinuation
    from model_runtime.runs.errors import (
        InvalidCheckpointContinuation,
        InvalidRunPlan,
        InvalidRunRequest,
        PlanTooLarge,
        RunPlanExecutionError,
        RunsError,
    )
    from model_runtime.runs.execution import execute_runs
    from model_runtime.runs.experiment import ExperimentBase
    from model_runtime.runs.json_values import (
        NonFiniteJsonValue,
        NonFiniteJsonValueError,
        non_finite_json_values,
        replace_non_finite_json,
        require_finite_json,
    )
    from model_runtime.runs.planning import accept_run_plan, plan_runs
    from model_runtime.runs.progress import JsonlRunProgress, RunProgress
    from model_runtime.runs.records import (
        PlanningBudget,
        PresetSearch,
        RandomSource,
        RunParameter,
        RunPlan,
        RunPlanRetry,
        RunRequest,
        RunResult,
        RunSpec,
        SearchAxisSelection,
        SearchSpec,
        SubmittedRun,
    )

__all__ = [
    "CheckpointContinuation",
    "CheckpointAdmissionPolicy",
    "DEFAULT_CHECKPOINT_ADMISSION_POLICY",
    "InvalidCheckpointContinuation",
    "InvalidRunPlan",
    "InvalidRunRequest",
    "FilesystemRunArtifacts",
    "JsonlRunProgress",
    "NonFiniteJsonValue",
    "NonFiniteJsonValueError",
    "PlanTooLarge",
    "PlanningBudget",
    "PresetSearch",
    "RandomSource",
    "RunParameter",
    "RunPlan",
    "RunPlanExecutionError",
    "RunPlanRetry",
    "RunArtifacts",
    "RunProgress",
    "RunRequest",
    "RunResult",
    "RunSpec",
    "RunsError",
    "SearchAxisSelection",
    "SearchSpec",
    "SubmittedRun",
    "accept_run_plan",
    "execute_runs",
    "ExperimentBase",
    "plan_runs",
    "non_finite_json_values",
    "replace_non_finite_json",
    "require_finite_json",
]

_ERROR_EXPORTS = {
    "InvalidCheckpointContinuation",
    "InvalidRunPlan",
    "InvalidRunRequest",
    "PlanTooLarge",
    "RunPlanExecutionError",
    "RunsError",
}
_CHECKPOINT_EXPORTS = {"CheckpointContinuation"}
_CHECKPOINT_ADMISSION_EXPORTS = {
    "CheckpointAdmissionPolicy",
    "DEFAULT_CHECKPOINT_ADMISSION_POLICY",
}
_ARTIFACT_EXPORTS = {"FilesystemRunArtifacts", "RunArtifacts"}
_PROGRESS_EXPORTS = {"JsonlRunProgress", "RunProgress"}
_JSON_VALUE_EXPORTS = {
    "NonFiniteJsonValue",
    "NonFiniteJsonValueError",
    "non_finite_json_values",
    "replace_non_finite_json",
    "require_finite_json",
}
_RECORD_EXPORTS = {
    "PlanningBudget",
    "PresetSearch",
    "RandomSource",
    "RunParameter",
    "RunPlan",
    "RunPlanRetry",
    "RunRequest",
    "RunResult",
    "RunSpec",
    "SearchAxisSelection",
    "SearchSpec",
    "SubmittedRun",
}
_PLANNING_EXPORTS = {"accept_run_plan", "plan_runs"}
_EXECUTION_EXPORTS = {"execute_runs"}
_EXPERIMENT_EXPORTS = {"ExperimentBase"}

_EXPORT_MODULES = {
    **dict.fromkeys(_CHECKPOINT_EXPORTS, "model_runtime.runs.checkpoints"),
    **dict.fromkeys(
        _CHECKPOINT_ADMISSION_EXPORTS,
        "model_runtime.runs.checkpoint_admission",
    ),
    **dict.fromkeys(_ERROR_EXPORTS, "model_runtime.runs.errors"),
    **dict.fromkeys(_ARTIFACT_EXPORTS, "model_runtime.runs.artifacts"),
    **dict.fromkeys(_PROGRESS_EXPORTS, "model_runtime.runs.progress"),
    **dict.fromkeys(_JSON_VALUE_EXPORTS, "model_runtime.runs.json_values"),
    **dict.fromkeys(_RECORD_EXPORTS, "model_runtime.runs.records"),
    **dict.fromkeys(_PLANNING_EXPORTS, "model_runtime.runs.planning"),
    **dict.fromkeys(_EXECUTION_EXPORTS, "model_runtime.runs.execution"),
    **dict.fromkeys(_EXPERIMENT_EXPORTS, "model_runtime.runs.experiment"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
