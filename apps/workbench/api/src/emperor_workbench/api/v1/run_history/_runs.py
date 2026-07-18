from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import (
    get_project_adapter_client,
    get_run_history_service,
    get_workbench_settings,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
    run_mutation_io,
)
from emperor_workbench.api.v1.run_history._contracts import (
    DEFAULT_LOG_PAGE_LIMIT,
    MAX_LOG_PAGE_LIMIT,
    LogCheckpointsRequest,
    LogCheckpointsResponse,
    LogPresetDeleteRequest,
    LogRunArtifactsResponse,
    LogRunDeleteFiltersRequest,
    LogRunDeletePlanResponse,
    LogRunDeleteResponse,
    LogRunsResponse,
)
from emperor_workbench.api.v1.run_history._mapping import (
    log_checkpoints_to_payload,
    log_run_artifacts_to_payload,
    log_run_delete_plan_to_payload,
    log_run_delete_result_to_payload,
    log_run_page_to_payload,
)
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import (
    LogRunDeletePlan,
    LogRunDeleteResult,
    RunHistoryFailure,
    RunHistoryService,
)
from emperor_workbench.settings import WorkbenchApiSettings

router = APIRouter()


def _model_query_ids(
    model_types: list[str] | None,
    models: list[str] | None,
    *,
    project_adapter: ProjectAdapterClient,
) -> list[str] | None:
    if not model_types:
        return models
    if not models:
        raise RunHistoryFailure("Log model filters require modelType and model.")
    if len(model_types) != len(models):
        raise RunHistoryFailure("Log modelType and model filters must be paired.")
    catalog = ModelPackageCatalog(project_adapter)
    return [
        catalog.require_id(model_type, model)
        for model_type, model in zip(model_types, models, strict=True)
    ]


def _model_filter_ids(
    request: LogRunDeleteFiltersRequest,
    *,
    project_adapter: ProjectAdapterClient,
) -> list[str]:
    catalog = ModelPackageCatalog(project_adapter)
    return [
        catalog.require_id(model.modelType, model.model) for model in request.models
    ]


@router.get(
    "/runs",
    response_model=LogRunsResponse,
    summary="List log runs",
    response_description="Historical TensorBoard runs indexed from the logs root.",
)
async def logs_runs(
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
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
    model_ids = await run_blocking_io(
        _model_query_ids,
        modelType,
        model,
        project_adapter=project_adapter,
    )
    page = await run_blocking_io(
        service.list_runs,
        limit=limit,
        offset=offset,
        experiment=experiment,
        model=model_ids,
        preset=preset,
        dataset=dataset,
        experiment_task=experimentTask,
        has_event_files=has_event_files,
        projection=projection,
    )
    return LogRunsResponse.model_validate(log_run_page_to_payload(page))


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
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> LogRunDeletePlanResponse:
    def create_delete_plan() -> LogRunDeletePlan:
        return service.create_delete_plan(
            experiments=request.experiments,
            datasets=request.datasets,
            models=_model_filter_ids(
                request,
                project_adapter=project_adapter,
            ),
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
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> LogRunDeleteResponse:
    def delete_runs() -> LogRunDeleteResult:
        return service.delete_runs(
            experiments=request.experiments,
            datasets=request.datasets,
            models=_model_filter_ids(
                request,
                project_adapter=project_adapter,
            ),
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
    def create_delete_plan() -> LogRunDeletePlan:
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
    def delete_preset() -> LogRunDeleteResult:
        return service.delete_preset(
            experiment=request.experiment,
            preset=request.preset,
        )

    return LogRunDeleteResponse.model_validate(
        log_run_delete_result_to_payload(await run_mutation_io(delete_preset))
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
    return LogRunArtifactsResponse.model_validate(log_run_artifacts_to_payload(details))


__all__ = ["router"]
