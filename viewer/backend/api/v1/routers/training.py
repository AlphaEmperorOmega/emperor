"""Training job lifecycle endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_training_job_service
from viewer.backend.schemas import (
    MonitorDataResponse,
    ParameterStatusResponse,
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingProgressEventsResponse,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from viewer.backend.services.training import TrainingJobService
from viewer.backend.training_contracts import (
    CreateTrainingJobCommand,
    CreateTrainingRunPlanCommand,
    TrainingRunPlanView,
    TrainingSearch,
)

router = APIRouter(
    prefix="/training",
    tags=["training"],
    dependencies=[Depends(require_bearer_auth)],
)


@router.post(
    "/jobs",
    response_model=TrainingJobResponse,
    summary="Create a training job",
    response_description="Created training job state.",
)
async def create_training_job(
    request: TrainingJobCreateRequest,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
) -> TrainingJobResponse:
    command = CreateTrainingJobCommand(
        model=request.model,
        preset=request.preset,
        presets=request.presets,
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        monitors=request.monitors,
        search=(
            TrainingSearch.from_payload(request.search.model_dump())
            if request.search is not None
            else None
        ),
        run_plan=(
            TrainingRunPlanView.from_payload(request.runPlan.model_dump())
            if request.runPlan is not None
            else None
        ),
    )
    return TrainingJobResponse.model_validate(
        service.create_job(command).to_api_payload()
    )


@router.post(
    "/run-plan",
    response_model=TrainingRunPlanResponse,
    summary="Create a training run plan",
    response_description="Materialized training runs for the current request.",
)
async def create_training_run_plan(
    request: TrainingRunPlanCreateRequest,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
) -> TrainingRunPlanResponse:
    command = CreateTrainingRunPlanCommand(
        model=request.model,
        preset=request.preset,
        presets=request.presets,
        datasets=request.datasets,
        overrides=request.overrides,
        log_folder=request.logFolder,
        search=(
            TrainingSearch.from_payload(request.search.model_dump())
            if request.search is not None
            else None
        ),
    )
    return TrainingRunPlanResponse.model_validate(
        service.create_run_plan(command).to_api_payload()
    )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get a training job",
    response_description="Current training job state.",
)
async def training_job(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
) -> TrainingJobResponse:
    return TrainingJobResponse.model_validate(
        service.get_job(job_id).to_api_payload()
    )


@router.get(
    "/jobs/{job_id}/events",
    response_model=TrainingProgressEventsResponse,
    summary="Read training progress event history",
    response_description="Paginated raw training progress events.",
)
async def training_job_events(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> TrainingProgressEventsResponse:
    return TrainingProgressEventsResponse.model_validate(
        service.get_job_events(
            job_id,
            offset=offset,
            limit=limit,
        )
    )


@router.get(
    "/jobs/{job_id}/monitor-data",
    response_model=MonitorDataResponse,
    summary="Read training monitor data",
    response_description="Monitor scalars, histograms, and images for a training job.",
)
async def training_job_monitor_data(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    node_path: str = Query(..., alias="nodePath"),
    dataset: str | None = None,
    preset: str | None = None,
) -> MonitorDataResponse:
    return MonitorDataResponse.model_validate(
        service.get_monitor_data(
            job_id,
            node_path=node_path,
            dataset=dataset,
            preset=preset,
        )
    )


@router.get(
    "/jobs/{job_id}/monitor-parameter-status",
    response_model=ParameterStatusResponse,
    summary="Read training monitor parameter status",
    response_description="Weight and bias update status for a training job.",
)
async def training_job_monitor_parameter_status(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    dataset: str | None = None,
    preset: str | None = None,
) -> ParameterStatusResponse:
    return ParameterStatusResponse.model_validate(
        service.get_parameter_status(
            job_id,
            dataset=dataset,
            preset=preset,
        )
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=TrainingJobResponse,
    summary="Cancel a training job",
    response_description="Cancelled training job state.",
)
async def cancel_training_job(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
) -> TrainingJobResponse:
    return TrainingJobResponse.model_validate(
        service.cancel_job(job_id).to_api_payload()
    )
