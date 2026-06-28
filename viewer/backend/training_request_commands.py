"""Translate public training request schemas into backend commands."""

from __future__ import annotations

from viewer.backend.model_identity import require_model_id
from viewer.backend.schemas import (
    TrainingJobCreateRequest,
    TrainingRunPlanCreateRequest,
)
from viewer.backend.training_contracts import (
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
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        monitors=request.monitors,
        search=_search_from_request(request),
    )


def create_training_job_command(
    request: TrainingJobCreateRequest,
) -> CreateTrainingJobCommand:
    return CreateTrainingJobCommand(
        model=require_model_id(request.modelType, request.model),
        preset=request.preset,
        presets=request.presets,
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        monitors=request.monitors,
        search=_search_from_request(request),
        run_plan=(
            TrainingRunPlanView.from_payload(request.runPlan.model_dump())
            if request.runPlan is not None
            else None
        ),
    )


__all__ = ["create_run_plan_command", "create_training_job_command"]
