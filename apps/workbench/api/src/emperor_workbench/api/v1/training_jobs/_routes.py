from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import (
    get_project_adapter_client,
    get_training_job_service,
    get_workbench_settings,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
    run_mutation_io,
)
from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1._monitoring_contracts import (
    MonitorDataResponse,
    ParameterStatusResponse,
)
from emperor_workbench.api.v1.run_history._mapping import (
    monitor_data_to_payload,
    parameter_status_to_payload,
)
from emperor_workbench.api.v1.training_jobs._commands import (
    create_training_job_command,
)
from emperor_workbench.api.v1.training_jobs._contracts import (
    TrainingJobCreateRequest,
    TrainingJobReconcileRequest,
    TrainingJobResponse,
    TrainingProgressEventsResponse,
)
from emperor_workbench.api.v1.training_jobs._mapping import (
    training_events_page_to_payload,
    training_job_to_payload,
)
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService

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
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def create_training_job(
    request: TrainingJobCreateRequest,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> TrainingJobResponse:
    command = await run_blocking_io(
        create_training_job_command,
        request,
        project_adapter=project_adapter,
    )
    return TrainingJobResponse.model_validate(
        training_job_to_payload(
            await run_mutation_io(
                service.create_job,
                command,
            )
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
    return TrainingJobResponse.model_validate(
        training_job_to_payload(await run_blocking_io(service.get_job, job_id))
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
        training_events_page_to_payload(
            await run_blocking_io(
                service.get_job_events,
                job_id,
                offset=offset,
                limit=limit,
            )
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
    node_path: Annotated[str, Query(alias="nodePath")],
    dataset: str | None = None,
    preset: str | None = None,
) -> MonitorDataResponse:
    return MonitorDataResponse.model_validate(
        monitor_data_to_payload(
            await run_blocking_io(
                service.get_monitor_data,
                job_id,
                node_path=node_path,
                dataset=dataset,
                preset=preset,
            )
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
        parameter_status_to_payload(
            await run_blocking_io(
                service.get_parameter_status,
                job_id,
                dataset=dataset,
                preset=preset,
            )
        )
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=TrainingJobResponse,
    summary="Cancel a training job",
    response_description="Cancelled training job state.",
)
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def cancel_training_job(
    job_id: str,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> TrainingJobResponse:
    return TrainingJobResponse.model_validate(
        training_job_to_payload(await run_mutation_io(service.cancel_job, job_id))
    )


@router.post(
    "/jobs/{job_id}/reconcile",
    response_model=TrainingJobResponse,
    summary="Reconcile an unknown training job",
    response_description="Operator-reconciled training job state.",
)
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def reconcile_training_job(
    job_id: str,
    request: TrainingJobReconcileRequest,
    service: Annotated[TrainingJobService, Depends(get_training_job_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> TrainingJobResponse:
    return TrainingJobResponse.model_validate(
        training_job_to_payload(
            await run_mutation_io(
                service.reconcile_job,
                job_id,
                action=request.action,
                reason=request.reason,
            )
        )
    )
