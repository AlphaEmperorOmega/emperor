"""Translate public training request schemas into backend commands."""

from __future__ import annotations

from workbench.backend.model_identity import require_model_id
from workbench.backend.schemas import (
    TrainingJobCreateRequest,
    TrainingRunPlanCreateRequest,
)
from workbench.backend.training_contracts import (
    CreateTrainingJobCommand,
    CreateTrainingRunPlanCommand,
    TrainingRunPlanView,
    TrainingSearch,
)


def _search_from_request(
    request: TrainingJobCreateRequest | TrainingRunPlanCreateRequest,
) -> TrainingSearch | None:
    if request.search is None:
        return None
    return TrainingSearch.from_payload(request.search.model_dump())


def create_run_plan_command(
    request: TrainingRunPlanCreateRequest,
) -> CreateTrainingRunPlanCommand:
    return CreateTrainingRunPlanCommand(
        model=require_model_id(request.modelType, request.model),
        preset=request.preset,
        presets=request.presets,
        experiment_task=request.experimentTask,
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        monitors=request.monitors,
        search=_search_from_request(request),
    )


def create_training_job_command(
    request: TrainingJobCreateRequest,
) -> CreateTrainingJobCommand:
    run_plan = (
        TrainingRunPlanView.from_payload(request.runPlan.model_dump())
        if request.runPlan is not None
        else None
    )
    return CreateTrainingJobCommand(
        model=require_model_id(request.modelType, request.model),
        preset=request.preset,
        presets=request.presets,
        experiment_task=(
            request.experimentTask
            if request.experimentTask is not None
            else (run_plan.experiment_task if run_plan is not None else None)
        ),
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        monitors=request.monitors,
        search=_search_from_request(request),
        run_plan=run_plan,
    )


__all__ = ["create_run_plan_command", "create_training_job_command"]
