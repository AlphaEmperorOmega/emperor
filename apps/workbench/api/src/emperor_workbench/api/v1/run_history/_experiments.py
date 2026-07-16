from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import (
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
    LogExperimentDeleteResponse,
    LogExperimentsResponse,
)
from emperor_workbench.api.v1.run_history._mapping import (
    log_experiment_delete_to_payload,
    log_experiment_page_to_payload,
)
from emperor_workbench.run_history import RunHistoryService
from emperor_workbench.settings import WorkbenchApiSettings

router = APIRouter()


@router.get(
    "/experiments",
    response_model=LogExperimentsResponse,
    summary="List log experiments",
    response_description="Log experiment folders indexed from the logs root.",
)
async def logs_experiments(
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    limit: Annotated[int, Query(ge=1, le=MAX_LOG_PAGE_LIMIT)] = (
        DEFAULT_LOG_PAGE_LIMIT
    ),
    offset: Annotated[int, Query(ge=0)] = 0,
) -> LogExperimentsResponse:
    page = await run_blocking_io(service.list_experiments, limit=limit, offset=offset)
    return LogExperimentsResponse.model_validate(log_experiment_page_to_payload(page))


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


__all__ = ["router"]
