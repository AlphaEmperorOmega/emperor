"""Training job lifecycle endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_training_job_service
from viewer.backend.schemas import (
    MonitorDataResponse,
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from viewer.backend.services.training import TrainingJobService

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
    return TrainingJobResponse.model_validate(
        service.create_job(
            model=request.model,
            preset=request.preset,
            presets=request.presets,
            datasets=request.datasets,
            overrides=request.overrides,
            log_folder=request.logFolder,
            monitors=request.monitors,
            search=request.search,
            run_plan=request.runPlan.model_dump() if request.runPlan is not None else None,
        )
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
    return TrainingRunPlanResponse.model_validate(
        service.create_run_plan(
            model=request.model,
            preset=request.preset,
            presets=request.presets,
            datasets=request.datasets,
            overrides=request.overrides,
            log_folder=request.logFolder,
            search=request.search,
        )
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
    return TrainingJobResponse.model_validate(service.get_job(job_id))


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
    return TrainingJobResponse.model_validate(service.cancel_job(job_id))
