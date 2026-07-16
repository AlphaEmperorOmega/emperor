from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import get_run_history_service
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
)
from emperor_workbench.api.v1.run_history._contracts import (
    LogMediaRequest,
    LogMediaResponse,
    LogParameterStatusRequest,
    LogParameterStatusResponse,
    LogScalarsRequest,
    LogScalarsResponse,
    LogTagsRequest,
    LogTagsResponse,
    MonitorDataResponse,
)
from emperor_workbench.api.v1.run_history._mapping import (
    log_media_to_payload,
    log_run_tags_to_payload,
    log_scalar_series_to_payload,
    monitor_data_to_payload,
    parameter_status_to_payload,
)
from emperor_workbench.run_history import RunHistoryService

router = APIRouter()


@router.post(
    "/tags",
    response_model=LogTagsResponse,
    summary="Read log-run tags",
    response_description=(
        "Scalar, histogram, image, and text tags for requested runs."
    ),
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def logs_tags(
    request: LogTagsRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogTagsResponse:
    tags_for_runs = await run_blocking_io(service.tags_for_runs, request.runIds)
    return LogTagsResponse.model_validate(
        {"runs": [log_run_tags_to_payload(tags) for tags in tags_for_runs]}
    )


@router.post(
    "/scalars",
    response_model=LogScalarsResponse,
    summary="Read log-run scalar series",
    response_description="Requested scalar series from historical TensorBoard runs.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def logs_scalars(
    request: LogScalarsRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogScalarsResponse:
    scalar_series = await run_blocking_io(
        service.scalars_for_runs,
        run_ids=request.runIds,
        tags=request.tags,
        max_points=request.maxPoints,
        sampling=request.sampling,
    )
    return LogScalarsResponse.model_validate(
        {"series": [log_scalar_series_to_payload(series) for series in scalar_series]}
    )


@router.post(
    "/media",
    response_model=LogMediaResponse,
    summary="Read log-run media summaries",
    response_description="Requested TensorBoard image and text summaries.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def logs_media(
    request: LogMediaRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogMediaResponse:
    media = await run_blocking_io(
        service.media_for_runs,
        run_ids=request.runIds,
        image_tags=request.imageTags,
        text_tags=request.textTags,
    )
    return LogMediaResponse.model_validate(log_media_to_payload(media))


@router.post(
    "/parameter-status",
    response_model=LogParameterStatusResponse,
    summary="Read log-run parameter status",
    response_description="Weight and bias update status for requested historical runs.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def logs_parameter_status(
    request: LogParameterStatusRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogParameterStatusResponse:
    statuses = await run_blocking_io(
        service.parameter_status_for_runs,
        request.runIds,
    )
    return LogParameterStatusResponse(
        runs=[parameter_status_to_payload(status) for status in statuses]
    )


@router.get(
    "/runs/{run_id}/monitor-data",
    response_model=MonitorDataResponse,
    summary="Read historical monitor data",
    response_description="Monitor scalars, histograms, and images for a log run.",
)
async def log_run_monitor_data(
    run_id: str,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    node_path: Annotated[str, Query(alias="nodePath")],
) -> MonitorDataResponse:
    data = await run_blocking_io(
        service.monitor_data_for_run,
        run_id,
        node_path=node_path,
    )
    return MonitorDataResponse.model_validate(monitor_data_to_payload(data))


__all__ = ["router"]
