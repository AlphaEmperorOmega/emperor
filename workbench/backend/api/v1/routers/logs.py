"""TensorBoard archive and historical log-run endpoints."""

from __future__ import annotations

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, BinaryIO, Literal

from fastapi import APIRouter, Depends, Query, Request

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    declare_http_operation,
)
from workbench.backend.api.v1.log_archive_upload import (
    parse_multipart_log_archive_upload,
)
from workbench.backend.api.v1.logs_mapping import (
    log_archive_import_to_payload,
    log_checkpoints_to_payload,
    log_experiment_delete_to_payload,
    log_experiment_page_to_payload,
    log_media_to_payload,
    log_monitor_data_to_payload,
    log_parameter_status_to_payload,
    log_run_artifacts_to_payload,
    log_run_delete_plan_to_payload,
    log_run_delete_result_to_payload,
    log_run_page_to_payload,
    log_run_tags_to_payload,
    log_scalar_series_to_payload,
)
from workbench.backend.blocking import (
    BLOCKING_WORK_TIMEOUT_MESSAGE,
    DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
    named_blocking_work_limiter,
    run_blocking_io,
)
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.errors import ApiError
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.dependencies import (
    get_run_history_service,
    get_workbench_settings,
)
from workbench.backend.model_identity import require_model_id
from workbench.backend.mutation_execution import run_mutation_io
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history.errors import RunHistoryFailure
from workbench.backend.schemas import (
    LogArchiveImportResponse,
    LogCheckpointsRequest,
    LogCheckpointsResponse,
    LogExperimentDeleteResponse,
    LogExperimentsResponse,
    LogMediaRequest,
    LogMediaResponse,
    LogParameterStatusRequest,
    LogParameterStatusResponse,
    LogPresetDeleteRequest,
    LogRunArtifactsResponse,
    LogRunDeleteFiltersRequest,
    LogRunDeletePlanResponse,
    LogRunDeleteResponse,
    LogRunsResponse,
    LogScalarsRequest,
    LogScalarsResponse,
    LogTagsRequest,
    LogTagsResponse,
    MonitorDataResponse,
)

router = APIRouter(
    prefix="/logs",
    tags=["logs"],
    dependencies=[Depends(require_bearer_auth)],
)
# Read endpoints are async at the API boundary so ASGI clients do not rely on
# FastAPI sync-route dispatch. Mutations coordinate Log Experiment ownership in
# the app-scoped capability layer.
DEFAULT_LOG_PAGE_LIMIT = 500
MAX_LOG_PAGE_LIMIT = 2000
LOG_ARCHIVE_UPLOAD_MEMORY_SPOOL_SIZE = 1024 * 1024
LOG_ARCHIVE_UPLOAD_LIMITER_NAME = "log-archive-upload"


def _upload_too_large_error(limit: int) -> ApiError:
    return ApiError(
        f"Log archive upload exceeds the {limit} byte limit.",
        status_code=413,
    )


async def _read_upload_body_with_limit(
    request: Request,
    *,
    max_upload_size: int | None,
) -> BinaryIO:
    body = tempfile.SpooledTemporaryFile(
        max_size=LOG_ARCHIVE_UPLOAD_MEMORY_SPOOL_SIZE,
        mode="w+b",
    )
    total_size = 0
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="workbench-upload-spool",
    )

    async def run_file_call(callable_object, *args):
        future = executor.submit(callable_object, *args)
        while not future.done():
            await asyncio.sleep(0)
        return future.result()

    try:
        async for chunk in request.stream():
            total_size += len(chunk)
            if max_upload_size is not None and total_size > max_upload_size:
                raise _upload_too_large_error(max_upload_size)
            await run_file_call(body.write, chunk)
        await run_file_call(body.seek, 0)
        return body
    except BaseException:
        await run_file_call(body.close)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def _model_query_ids(
    model_types: list[str] | None,
    models: list[str] | None,
) -> list[str] | None:
    if not model_types:
        return models
    if not models:
        raise RunHistoryFailure("Log model filters require modelType and model.")
    if len(model_types) != len(models):
        raise RunHistoryFailure("Log modelType and model filters must be paired.")
    return [
        require_model_id(model_type, model)
        for model_type, model in zip(model_types, models, strict=True)
    ]


def _model_filter_ids(request: LogRunDeleteFiltersRequest) -> list[str]:
    return [require_model_id(model.modelType, model.model) for model in request.models]


@router.get(
    "/runs",
    response_model=LogRunsResponse,
    summary="List log runs",
    response_description="Historical TensorBoard runs indexed from the logs root.",
)
async def logs_runs(
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_LOG_PAGE_LIMIT),
    ] = DEFAULT_LOG_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
    experiment: Annotated[list[str] | None, Query()] = None,
    modelType: Annotated[list[str] | None, Query()] = None,
    model: Annotated[list[str] | None, Query()] = None,
    preset: Annotated[list[str] | None, Query()] = None,
    dataset: Annotated[list[str] | None, Query()] = None,
    experimentTask: Annotated[str | None, Query()] = None,
    has_event_files: Annotated[
        bool | None,
        Query(alias="hasEventFiles"),
    ] = None,
    projection: Annotated[Literal["full", "summary"], Query()] = "full",
) -> LogRunsResponse:
    page = await run_blocking_io(
            service.list_runs,
            limit=limit,
            offset=offset,
            experiment=experiment,
            model=_model_query_ids(modelType, model),
            preset=preset,
            dataset=dataset,
            experiment_task=experimentTask,
            has_event_files=has_event_files,
            projection=projection,
        )
    return LogRunsResponse.model_validate(log_run_page_to_payload(page))


@router.get(
    "/experiments",
    response_model=LogExperimentsResponse,
    summary="List log experiments",
    response_description="Log experiment folders indexed from the logs root.",
)
async def logs_experiments(
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    limit: int = Query(DEFAULT_LOG_PAGE_LIMIT, ge=1, le=MAX_LOG_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
) -> LogExperimentsResponse:
    page = await run_blocking_io(service.list_experiments, limit=limit, offset=offset)
    return LogExperimentsResponse.model_validate(
        log_experiment_page_to_payload(page)
    )


@router.post(
    "/import",
    response_model=LogArchiveImportResponse,
    summary="Import log archive",
    response_description="Extracted log archive import summary.",
)
@declare_http_operation(HttpOperationPolicy.LOG_IMPORT)
async def import_log_archive(
    request: Request,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> LogArchiveImportResponse:
    max_upload_size = settings.effective_max_upload_size
    max_extracted_size = settings.effective_max_log_archive_extracted_size
    max_member_count = settings.max_log_archive_member_count
    max_path_bytes = settings.max_log_archive_path_bytes
    upload_concurrency = settings.log_archive_upload_concurrency
    content_type = request.headers.get("content-type", "")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            upload_size = int(content_length)
        except ValueError:
            upload_size = 0
        if max_upload_size is not None and upload_size > max_upload_size:
            raise _upload_too_large_error(max_upload_size)

    upload_limiter = named_blocking_work_limiter(
        f"{LOG_ARCHIVE_UPLOAD_LIMITER_NAME}:{upload_concurrency}",
        upload_concurrency,
    )
    try:
        await asyncio.wait_for(
            upload_limiter.acquire(),
            DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise ApiError(
            BLOCKING_WORK_TIMEOUT_MESSAGE,
            status_code=503,
        ) from exc

    limiter_handed_to_worker = False
    try:
        body = await _read_upload_body_with_limit(
            request,
            max_upload_size=max_upload_size,
        )

        def parse_and_extract_archive():
            try:
                upload = parse_multipart_log_archive_upload(
                    content_type=content_type,
                    body=body,
                    max_upload_size=max_upload_size,
                )
                return service.import_archive(
                    archive=upload.content,
                    filename=upload.filename,
                    max_upload_size=max_upload_size,
                    max_extracted_size=max_extracted_size,
                    max_member_count=max_member_count,
                    max_path_bytes=max_path_bytes,
                )
            finally:
                body.close()

        limiter_handed_to_worker = True
        result = await run_mutation_io(
            parse_and_extract_archive,
            limiter=upload_limiter,
            limiter_already_acquired=True,
        )
        return LogArchiveImportResponse.model_validate(
            log_archive_import_to_payload(result)
        )
    finally:
        if not limiter_handed_to_worker:
            upload_limiter.release()


@router.post(
    "/checkpoints",
    response_model=LogCheckpointsResponse,
    summary="Read log-run checkpoints",
    response_description="Checkpoint file metadata for requested historical runs.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def logs_checkpoints(
    request: LogCheckpointsRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogCheckpointsResponse:
    checkpoints = await run_blocking_io(service.checkpoints_for_runs, request.runIds)
    return LogCheckpointsResponse.model_validate(
        log_checkpoints_to_payload(checkpoints)
    )


@router.delete(
    "/experiments/{experiment}",
    response_model=LogExperimentDeleteResponse,
    summary="Delete a log experiment",
    response_description="Deleted experiment metadata and removed run ids.",
)
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def delete_log_experiment(
    experiment: str,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> LogExperimentDeleteResponse:
    def delete_experiment():
        return service.delete_experiment(experiment)

    return LogExperimentDeleteResponse.model_validate(
        log_experiment_delete_to_payload(await run_mutation_io(delete_experiment))
    )


@router.post(
    "/runs/delete-plan",
    response_model=LogRunDeletePlanResponse,
    summary="Plan filtered log-run deletion",
    response_description="Matched run folders and active training-job blockers.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def log_run_delete_plan(
    request: LogRunDeleteFiltersRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogRunDeletePlanResponse:
    def create_delete_plan():
        return service.create_delete_plan(
            experiments=request.experiments,
            datasets=request.datasets,
            models=_model_filter_ids(request),
            presets=request.presets,
            run_ids=request.runIds,
        )

    return LogRunDeletePlanResponse.model_validate(
        log_run_delete_plan_to_payload(await run_blocking_io(create_delete_plan))
    )


@router.post(
    "/runs/delete",
    response_model=LogRunDeleteResponse,
    summary="Delete filtered log runs",
    response_description="Deleted run metadata for matched version folders.",
)
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def delete_log_runs(
    request: LogRunDeleteFiltersRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> LogRunDeleteResponse:
    def delete_runs():
        return service.delete_runs(
            experiments=request.experiments,
            datasets=request.datasets,
            models=_model_filter_ids(request),
            presets=request.presets,
            run_ids=request.runIds,
        )

    return LogRunDeleteResponse.model_validate(
        log_run_delete_result_to_payload(await run_mutation_io(delete_runs))
    )


@router.post(
    "/runs/preset-delete-plan",
    response_model=LogRunDeletePlanResponse,
    summary="Plan preset log-run deletion",
    response_description="Every run matching one experiment and preset.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def log_preset_delete_plan(
    request: LogPresetDeleteRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogRunDeletePlanResponse:
    def create_delete_plan():
        return service.create_preset_delete_plan(
            experiment=request.experiment,
            preset=request.preset,
        )

    return LogRunDeletePlanResponse.model_validate(
        log_run_delete_plan_to_payload(await run_blocking_io(create_delete_plan))
    )


@router.post(
    "/runs/preset-delete",
    response_model=LogRunDeleteResponse,
    summary="Delete preset log runs",
    response_description="Deleted metadata for every matching run.",
)
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def delete_log_preset(
    request: LogPresetDeleteRequest,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> LogRunDeleteResponse:
    def delete_preset():
        return service.delete_preset(
            experiment=request.experiment,
            preset=request.preset,
        )

    return LogRunDeleteResponse.model_validate(
        log_run_delete_result_to_payload(await run_mutation_io(delete_preset))
    )


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
        {
            "series": [
                log_scalar_series_to_payload(series) for series in scalar_series
            ]
        }
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
        runs=[
            log_parameter_status_to_payload(status)
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
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
) -> LogRunArtifactsResponse:
    details = await run_blocking_io(service.artifacts_for_run, run_id)
    return LogRunArtifactsResponse.model_validate(
        log_run_artifacts_to_payload(details)
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
    node_path: str = Query(..., alias="nodePath"),
) -> MonitorDataResponse:
    data = await run_blocking_io(
            service.monitor_data_for_run,
            run_id,
            node_path=node_path,
        )
    return MonitorDataResponse.model_validate(log_monitor_data_to_payload(data))
