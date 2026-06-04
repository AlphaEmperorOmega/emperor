"""TensorBoard archive and historical log-run endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.dependencies import get_log_run_service, get_training_job_service
from viewer.backend.schemas import (
    LogExperimentDeleteResponse,
    LogExperimentResponse,
    LogExperimentsResponse,
    LogRunDeleteFiltersRequest,
    LogRunDeletePlanResponse,
    LogRunDeleteResponse,
    LogRunResponse,
    LogRunsResponse,
    LogRunTagsResponse,
    LogScalarSeriesResponse,
    LogScalarsRequest,
    LogScalarsResponse,
    LogTagsRequest,
    LogTagsResponse,
    MonitorDataResponse,
)
from viewer.backend.services.logs import LogRunService
from viewer.backend.services.training import TrainingJobService

router = APIRouter(prefix="/logs", tags=["logs"])
DEFAULT_LOG_PAGE_LIMIT = 500
MAX_LOG_PAGE_LIMIT = 2000


@router.get(
    "/runs",
    response_model=LogRunsResponse,
    summary="List log runs",
    response_description="Historical TensorBoard runs indexed from the logs root.",
)
async def logs_runs(
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    limit: int = Query(DEFAULT_LOG_PAGE_LIMIT, ge=1, le=MAX_LOG_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> LogRunsResponse:
    page = service.list_runs(limit=limit, offset=offset)
    return LogRunsResponse(
        total=page["total"],
        limit=page["limit"],
        offset=page["offset"],
        hasMore=page["hasMore"],
        runs=[LogRunResponse.model_validate(run) for run in page["runs"]],
    )


@router.get(
    "/experiments",
    response_model=LogExperimentsResponse,
    summary="List log experiments",
    response_description="Log experiment folders indexed from the logs root.",
)
async def logs_experiments(
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    limit: int = Query(DEFAULT_LOG_PAGE_LIMIT, ge=1, le=MAX_LOG_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> LogExperimentsResponse:
    page = service.list_experiments(limit=limit, offset=offset)
    return LogExperimentsResponse(
        total=page["total"],
        limit=page["limit"],
        offset=page["offset"],
        hasMore=page["hasMore"],
        experiments=[
            LogExperimentResponse.model_validate(experiment)
            for experiment in page["experiments"]
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
) -> LogExperimentDeleteResponse:
    return LogExperimentDeleteResponse.model_validate(
        service.delete_experiment(experiment)
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
            active_jobs=training_service.active_jobs(),
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
            active_jobs=training_service.active_jobs(),
        )
    )


@router.post(
    "/tags",
    response_model=LogTagsResponse,
    summary="Read log-run tags",
    response_description="Scalar, histogram, and image tags for requested runs.",
)
async def logs_tags(
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
async def logs_scalars(
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


@router.get(
    "/runs/{run_id}/monitor-data",
    response_model=MonitorDataResponse,
    summary="Read historical monitor data",
    response_description="Monitor scalars, histograms, and images for a log run.",
)
async def log_run_monitor_data(
    run_id: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    node_path: str = Query(..., alias="nodePath"),
) -> MonitorDataResponse:
    return MonitorDataResponse.model_validate(
        service.monitor_data_for_run(run_id, node_path=node_path)
    )
