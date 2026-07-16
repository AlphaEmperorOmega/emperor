from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from emperor_workbench.api._blocking import (
    named_blocking_work_limiter,
    run_blocking_io,
)
from emperor_workbench.api._dependencies import (
    get_inspection_service,
    get_project_adapter_client,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
)
from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1.inspection._contracts import (
    InspectRequest,
    InspectResponse,
)
from emperor_workbench.api.v1.inspection._mapping import inspection_response
from emperor_workbench.inspection import InspectionFailure, InspectionService
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
)
from emperor_workbench.project_adapter import ProjectAdapterClient

INSPECTION_BLOCKING_WORK_CONCURRENCY = 2
INSPECTION_BLOCKING_WORK_LIMITER_NAME = "inspection"

router = APIRouter(
    tags=["inspection"],
    dependencies=[Depends(require_bearer_auth)],
)


def _inspect_model(
    request: InspectRequest,
    *,
    service: InspectionService,
    project_adapter: ProjectAdapterClient,
):
    try:
        selected = ModelPackageCatalog(project_adapter).select_parts(
            request.modelType,
            request.model,
        )
    except ModelPackageFailure as exc:
        raise InspectionFailure(exc.detail, kind=exc.kind) from exc
    return service.inspect(
        selected,
        preset=request.preset,
        overrides=request.overrides,
        experiment_task=request.experimentTask,
        dataset=request.dataset,
        log_run_id=request.logRunId,
    )


@router.post(
    "/inspect",
    response_model=InspectResponse,
    summary="Inspect a model preset",
    response_description="Serialized model graph for the requested model preset.",
)
@declare_http_operation(HttpOperationPolicy.READ_ONLY)
async def inspect(
    request: InspectRequest,
    service: Annotated[InspectionService, Depends(get_inspection_service)],
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> InspectResponse:
    result = await run_blocking_io(
        _inspect_model,
        request,
        service=service,
        project_adapter=project_adapter,
        limiter=named_blocking_work_limiter(
            INSPECTION_BLOCKING_WORK_LIMITER_NAME,
            INSPECTION_BLOCKING_WORK_CONCURRENCY,
        ),
    )
    return inspection_response(result)


__all__ = ["router"]
