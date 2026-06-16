"""TensorBoard archive and historical log-run endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.blocking import run_blocking_io
from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_log_run_service, get_training_job_service
from viewer.backend.schemas import (
    LogCheckpointResponse,
    LogCheckpointsRequest,
    LogCheckpointsResponse,
    LogExperimentDeleteResponse,
    LogExperimentsResponse,
    LogImageSummaryResponse,
    LogMediaRequest,
    LogMediaResponse,
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
    LogTextSummaryResponse,
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
# Read endpoints are async at the API boundary so ASGI clients do not rely on
# FastAPI sync-route dispatch. Delete endpoints share mutable TrainingJobManager
# state with the training routes.
DEFAULT_LOG_PAGE_LIMIT = 500
MAX_LOG_PAGE_LIMIT = 2000
LOG_METADATA_RESPONSE_LIMIT = 500


def _bounded_metadata_response(items: list[object], *, label: str) -> dict[str, object]:
    returned = items[:LOG_METADATA_RESPONSE_LIMIT]
    truncated = len(items) > len(returned)
    return {
        "sourceItemCount": len(items),
        "returnedItemCount": len(returned),
        "truncated": truncated,
        "truncationReason": (
            f"{label} capped at {LOG_METADATA_RESPONSE_LIMIT} rows"
            if truncated
            else None
        ),
        "items": returned,
    }


def active_job_payloads(service: TrainingJobService) -> list[dict[str, str]]:
    return [job.to_api_payload() for job in service.active_jobs()]


@router.get(
    "/runs",
    response_model=LogRunsResponse,
    summary="List log runs",
    response_description="Historical TensorBoard runs indexed from the logs root.",
)
async def logs_runs(
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_LOG_PAGE_LIMIT),
    ] = DEFAULT_LOG_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    experiment: Annotated[list[str] | None, Query()] = None,
    model: Annotated[list[str] | None, Query()] = None,
    preset: Annotated[list[str] | None, Query()] = None,
    dataset: Annotated[list[str] | None, Query()] = None,
    has_event_files: Annotated[
        bool | None,
        Query(alias="hasEventFiles"),
    ] = None,
) -> LogRunsResponse:
    return LogRunsResponse.model_validate(
        await run_blocking_io(
            service.list_runs,
            limit=limit,
            offset=offset,
            experiment=experiment,
            model=model,
            preset=preset,
            dataset=dataset,
            has_event_files=has_event_files,
        )
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
    return LogExperimentsResponse.model_validate(
        await run_blocking_io(service.list_experiments, limit=limit, offset=offset)
    )


@router.post(
    "/checkpoints",
    response_model=LogCheckpointsResponse,
    summary="Read log-run checkpoints",
    response_description="Checkpoint file metadata for requested historical runs.",
)
async def logs_checkpoints(
    request: LogCheckpointsRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogCheckpointsResponse:
    checkpoints = await run_blocking_io(service.checkpoints_for_runs, request.runIds)
    bounded = _bounded_metadata_response(checkpoints, label="checkpoint metadata")
    return LogCheckpointsResponse(
        sourceItemCount=int(bounded["sourceItemCount"]),
        returnedItemCount=int(bounded["returnedItemCount"]),
        truncated=bool(bounded["truncated"]),
        truncationReason=(
            str(bounded["truncationReason"])
            if bounded["truncationReason"] is not None
            else None
        ),
        checkpoints=[
            LogCheckpointResponse.model_validate(checkpoint)
            for checkpoint in bounded["items"]
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
    response_description=(
        "Scalar, histogram, image, and text tags for requested runs."
    ),
)
async def logs_tags(
    request: LogTagsRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogTagsResponse:
    tags_for_runs = await run_blocking_io(service.tags_for_runs, request.runIds)
    return LogTagsResponse(
        runs=[
            LogRunTagsResponse.model_validate(tags)
            for tags in tags_for_runs
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
    scalar_series = await run_blocking_io(
        service.scalars_for_runs,
        run_ids=request.runIds,
        tags=request.tags,
        max_points=request.maxPoints,
        sampling=request.sampling,
    )
    return LogScalarsResponse(
        series=[
            LogScalarSeriesResponse.model_validate(series)
            for series in scalar_series
        ]
    )


@router.post(
    "/media",
    response_model=LogMediaResponse,
    summary="Read log-run media summaries",
    response_description="Requested TensorBoard image and text summaries.",
)
async def logs_media(
    request: LogMediaRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogMediaResponse:
    media = await run_blocking_io(
        service.media_for_runs,
        run_ids=request.runIds,
        image_tags=request.imageTags,
        text_tags=request.textTags,
    )
    return LogMediaResponse(
        eventBytes=media.get("eventBytes"),
        skippedEventFiles=media.get("skippedEventFiles"),
        sourceItemCount=media.get("sourceItemCount"),
        returnedItemCount=media.get("returnedItemCount"),
        truncated=media.get("truncated"),
        truncationReason=media.get("truncationReason"),
        images=[
            LogImageSummaryResponse.model_validate(image)
            for image in media["images"]
        ],
        texts=[
            LogTextSummaryResponse.model_validate(text)
            for text in media["texts"]
        ],
    )


@router.post(
    "/parameter-status",
    response_model=LogParameterStatusResponse,
    summary="Read log-run parameter status",
    response_description="Weight and bias update status for requested historical runs.",
)
async def logs_parameter_status(
    request: LogParameterStatusRequest,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogParameterStatusResponse:
    statuses = await run_blocking_io(
        service.parameter_status_for_runs,
        request.runIds,
    )
    return LogParameterStatusResponse(
        runs=[
            ParameterStatusResponse.model_validate(status)
            for status in statuses
        ]
    )


@router.get(
    "/runs/{run_id}/artifacts",
    response_model=LogRunArtifactsResponse,
    summary="Read historical run artifacts",
    response_description="Result, hparams, event, and checkpoint file metadata.",
)
async def log_run_artifacts(
    run_id: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
) -> LogRunArtifactsResponse:
    payload = await run_blocking_io(service.artifacts_for_run, run_id)
    return LogRunArtifactsResponse(
        runId=str(payload["runId"]),
        params=dict(payload["params"]),
        metrics=dict(payload["metrics"]),
        sourceItemCount=payload.get("sourceItemCount"),
        returnedItemCount=payload.get("returnedItemCount"),
        truncated=payload.get("truncated"),
        truncationReason=payload.get("truncationReason"),
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
async def log_run_monitor_data(
    run_id: str,
    service: Annotated[LogRunService, Depends(get_log_run_service)],
    node_path: str = Query(..., alias="nodePath"),
) -> MonitorDataResponse:
    return MonitorDataResponse.model_validate(
        await run_blocking_io(
            service.monitor_data_for_run,
            run_id,
            node_path=node_path,
        )
    )
