"""Training job lifecycle endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from workbench.backend.blocking import run_blocking_io
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import (
    require_bearer_auth,
    require_local_mutations_allowed,
)
from workbench.backend.dependencies import (
    get_training_job_service,
    get_workbench_settings,
)
from workbench.backend.schemas import (
    MonitorDataResponse,
    ParameterStatusResponse,
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingProgressEventsResponse,
    TrainingRunPlanCreateRequest,
    TrainingRunPlanResponse,
)
from workbench.backend.services.training import TrainingJobService
from workbench.backend.training_request_commands import (
    create_run_plan_command,
    create_training_job_command,
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
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> TrainingJobResponse:
    require_local_mutations_allowed(settings)
    return TrainingJobResponse.model_validate(
        (
            await run_blocking_io(
                service.create_job,
                create_training_job_command(request),
            )
        ).to_api_payload()
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
        (
            await run_blocking_io(
                service.create_run_plan,
                create_run_plan_command(request),
            )
        ).to_api_payload()
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
        (await run_blocking_io(service.get_job, job_id)).to_api_payload()
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
        await run_blocking_io(
            service.get_job_events,
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
        await run_blocking_io(
            service.get_monitor_data,
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
        await run_blocking_io(
            service.get_parameter_status,
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
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> TrainingJobResponse:
    require_local_mutations_allowed(settings)
    return TrainingJobResponse.model_validate(
        (await run_blocking_io(service.cancel_job, job_id)).to_api_payload()
    )
