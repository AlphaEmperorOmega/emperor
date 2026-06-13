"""TensorBoard archive and historical log-run endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_log_run_service, get_training_job_service
from viewer.backend.schemas import (
    LogCheckpointResponse,
    LogCheckpointsRequest,
    LogCheckpointsResponse,
    LogExperimentDeleteResponse,
    LogExperimentsResponse,
    LogParameterStatusRequest,
    LogParameterStatusResponse,
    LogRunArtifactResponse,
    LogRunArtifactsResponse,
    LogRunDeleteFiltersRequest,
    LogRunDeletePlanResponse,
    LogRunDeleteResponse,
    LogRunsResponse,
    LogRunTagsResponse,
    LogScalarSeriesResponse,
    LogScalarsRequest,
    LogScalarsResponse,
    LogTagsRequest,
    LogTagsResponse,
    MonitorDataResponse,
    ParameterStatusResponse,
)
from viewer.backend.services.logs import LogRunService
from viewer.backend.services.training import TrainingJobService

router = APIRouter(
    prefix="/logs",
    tags=["logs"],
    dependencies=[Depends(require_bearer_auth)],
)
# Read endpoints are sync handlers on purpose: FastAPI dispatches them to the
# worker threadpool, keeping TensorBoard reads and log scans off the event
# loop. Delete endpoints stay async because they share mutable
# TrainingJobManager state with the training routes.
DEFAULT_LOG_PAGE_LIMIT = 500
MAX_LOG_PAGE_LIMIT = 2000


def active_job_payloads(service: TrainingJobService) -> list[dict[str, str]]:
    return [job.to_api_payload() for job in service.active_jobs()]


@router.get(
    "/runs",
    response_model=LogRunsResponse,
    summary="List log runs",
    response_description="Historical TensorBoard runs indexed from the logs root.",
)
def logs_runs(
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    limit: int = Query(DEFAULT_LOG_PAGE_LIMIT, ge=1, le=MAX_LOG_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> LogRunsResponse:
    return LogRunsResponse.model_validate(
        service.list_runs(limit=limit, offset=offset)
    )


@router.get(
    "/experiments",
    response_model=LogExperimentsResponse,
    summary="List log experiments",
    response_description="Log experiment folders indexed from the logs root.",
)
def logs_experiments(
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    limit: int = Query(DEFAULT_LOG_PAGE_LIMIT, ge=1, le=MAX_LOG_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> LogExperimentsResponse:
    return LogExperimentsResponse.model_validate(
        service.list_experiments(limit=limit, offset=offset)
    )


@router.post(
    "/checkpoints",
    response_model=LogCheckpointsResponse,
    summary="Read log-run checkpoints",
    response_description="Checkpoint file metadata for requested historical runs.",
)
def logs_checkpoints(
    request: LogCheckpointsRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogCheckpointsResponse:
    return LogCheckpointsResponse(
        checkpoints=[
            LogCheckpointResponse.model_validate(checkpoint)
            for checkpoint in service.checkpoints_for_runs(request.runIds)
        ]
    )


@router.delete(
    "/experiments/{experiment}",
    response_model=LogExperimentDeleteResponse,
    summary="Delete a log experiment",
    response_description="Deleted experiment metadata and removed run ids.",
)
async def delete_log_experiment(
    experiment: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    training_service: Annotated[
        TrainingJobService,
        Depends(get_training_job_service),
    ],
) -> LogExperimentDeleteResponse:
    return LogExperimentDeleteResponse.model_validate(
        service.delete_experiment(
            experiment,
            active_jobs=active_job_payloads(training_service),
        )
    )


@router.post(
    "/runs/delete-plan",
    response_model=LogRunDeletePlanResponse,
    summary="Plan filtered log-run deletion",
    response_description="Matched run folders and active training-job blockers.",
)
async def log_run_delete_plan(
    request: LogRunDeleteFiltersRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    training_service: Annotated[
        TrainingJobService,
        Depends(get_training_job_service),
    ],
) -> LogRunDeletePlanResponse:
    return LogRunDeletePlanResponse.model_validate(
        service.create_delete_plan(
            experiments=request.experiments,
            datasets=request.datasets,
            models=request.models,
            presets=request.presets,
            run_ids=request.runIds,
            active_jobs=active_job_payloads(training_service),
        )
    )


@router.post(
    "/runs/delete",
    response_model=LogRunDeleteResponse,
    summary="Delete filtered log runs",
    response_description="Deleted run metadata for matched version folders.",
)
async def delete_log_runs(
    request: LogRunDeleteFiltersRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    training_service: Annotated[
        TrainingJobService,
        Depends(get_training_job_service),
    ],
) -> LogRunDeleteResponse:
    return LogRunDeleteResponse.model_validate(
        service.delete_runs(
            experiments=request.experiments,
            datasets=request.datasets,
            models=request.models,
            presets=request.presets,
            run_ids=request.runIds,
            active_jobs=active_job_payloads(training_service),
        )
    )


@router.post(
    "/tags",
    response_model=LogTagsResponse,
    summary="Read log-run tags",
    response_description="Scalar, histogram, and image tags for requested runs.",
)
def logs_tags(
    request: LogTagsRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogTagsResponse:
    return LogTagsResponse(
        runs=[
            LogRunTagsResponse.model_validate(tags)
            for tags in service.tags_for_runs(request.runIds)
        ]
    )


@router.post(
    "/scalars",
    response_model=LogScalarsResponse,
    summary="Read log-run scalar series",
    response_description="Requested scalar series from historical TensorBoard runs.",
)
def logs_scalars(
    request: LogScalarsRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogScalarsResponse:
    return LogScalarsResponse(
        series=[
            LogScalarSeriesResponse.model_validate(series)
            for series in service.scalars_for_runs(
                run_ids=request.runIds,
                tags=request.tags,
            )
        ]
    )


@router.post(
    "/parameter-status",
    response_model=LogParameterStatusResponse,
    summary="Read log-run parameter status",
    response_description="Weight and bias update status for requested historical runs.",
)
def logs_parameter_status(
    request: LogParameterStatusRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogParameterStatusResponse:
    return LogParameterStatusResponse(
        runs=[
            ParameterStatusResponse.model_validate(status)
            for status in service.parameter_status_for_runs(request.runIds)
        ]
    )


@router.get(
    "/runs/{run_id}/artifacts",
    response_model=LogRunArtifactsResponse,
    summary="Read historical run artifacts",
    response_description="Result, hparams, event, and checkpoint file metadata.",
)
def log_run_artifacts(
    run_id: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogRunArtifactsResponse:
    payload = service.artifacts_for_run(run_id)
    return LogRunArtifactsResponse(
        runId=str(payload["runId"]),
        params=dict(payload["params"]),
        metrics=dict(payload["metrics"]),
        artifacts=[
            LogRunArtifactResponse.model_validate(artifact)
            for artifact in payload["artifacts"]
        ],
        checkpoints=[
            LogCheckpointResponse.model_validate(checkpoint)
            for checkpoint in payload["checkpoints"]
        ],
    )


@router.get(
    "/runs/{run_id}/monitor-data",
    response_model=MonitorDataResponse,
    summary="Read historical monitor data",
    response_description="Monitor scalars, histograms, and images for a log run.",
)
def log_run_monitor_data(
    run_id: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    node_path: str = Query(..., alias="nodePath"),
) -> MonitorDataResponse:
    return MonitorDataResponse.model_validate(
        service.monitor_data_for_run(run_id, node_path=node_path)
    )
