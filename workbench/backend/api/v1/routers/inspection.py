"""Model inspection endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    declare_http_operation,
)
from workbench.backend.blocking import named_blocking_work_limiter, run_blocking_io
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.dependencies import get_inspection_service
from workbench.backend.schemas import (
    InspectRequest,
    InspectResponse,
)
from workbench.backend.services.inspection import InspectionService

INSPECTION_BLOCKING_WORK_CONCURRENCY = 2
INSPECTION_BLOCKING_WORK_LIMITER_NAME = "inspection"

router = APIRouter(
    tags=["inspection"],
    dependencies=[Depends(require_bearer_auth)],
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
) -> InspectResponse:
    return InspectResponse.model_validate(
        await run_blocking_io(
            service.inspect,
            model_type=request.modelType,
            model=request.model,
            preset=request.preset,
            overrides=request.overrides,
            experiment_task=request.experimentTask,
            dataset=request.dataset,
            log_run_id=request.logRunId,
            limiter=named_blocking_work_limiter(
                INSPECTION_BLOCKING_WORK_LIMITER_NAME,
                INSPECTION_BLOCKING_WORK_CONCURRENCY,
            ),
        )
    )
