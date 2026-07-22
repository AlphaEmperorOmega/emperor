from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_runtime.runs.artifacts import FilesystemRunArtifacts
    from model_runtime.runs.checkpoints import CheckpointContinuation
    from model_runtime.runs.errors import (
        InvalidCheckpointContinuation,
        InvalidRunPlan,
        InvalidRunRequest,
        PlanTooLarge,
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
    from model_runtime.runs.progress import JsonlTrainingProgressCallback
    from model_runtime.runs.records import (
        PlanningBudget,
        RandomSource,
        RunParameter,
        RunPlan,
        RunRequest,
        RunResult,
        RunSpec,
        SearchAxisSelection,
        SearchSpec,
        SubmittedRun,
    )

__all__ = [
    "CheckpointContinuation",
    "InvalidCheckpointContinuation",
    "InvalidRunPlan",
    "InvalidRunRequest",
    "FilesystemRunArtifacts",
    "JsonlTrainingProgressCallback",
    "NonFiniteJsonValue",
    "NonFiniteJsonValueError",
    "PlanTooLarge",
    "PlanningBudget",
    "RandomSource",
    "RunParameter",
    "RunPlan",
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
    "RunsError",
}
_CHECKPOINT_EXPORTS = {"CheckpointContinuation"}
_ARTIFACT_EXPORTS = {"FilesystemRunArtifacts"}
_PROGRESS_EXPORTS = {"JsonlTrainingProgressCallback"}
_JSON_VALUE_EXPORTS = {
    "NonFiniteJsonValue",
    "NonFiniteJsonValueError",
    "non_finite_json_values",
    "replace_non_finite_json",
    "require_finite_json",
}
_RECORD_EXPORTS = {
    "PlanningBudget",
    "RandomSource",
    "RunParameter",
    "RunPlan",
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


def __getattr__(name: str) -> Any:
    if name in _CHECKPOINT_EXPORTS:
        module_name = "model_runtime.runs.checkpoints"
    elif name in _ERROR_EXPORTS:
        module_name = "model_runtime.runs.errors"
    elif name in _ARTIFACT_EXPORTS:
        module_name = "model_runtime.runs.artifacts"
    elif name in _PROGRESS_EXPORTS:
        module_name = "model_runtime.runs.progress"
    elif name in _JSON_VALUE_EXPORTS:
        module_name = "model_runtime.runs.json_values"
    elif name in _RECORD_EXPORTS:
        module_name = "model_runtime.runs.records"
    elif name in _PLANNING_EXPORTS:
        module_name = "model_runtime.runs.planning"
    elif name in _EXECUTION_EXPORTS:
        module_name = "model_runtime.runs.execution"
    elif name in _EXPERIMENT_EXPORTS:
        module_name = "model_runtime.runs.experiment"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
