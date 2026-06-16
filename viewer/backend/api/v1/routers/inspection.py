"""Model inspection endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from viewer.backend.blocking import run_blocking_io
from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_inspection_service
from viewer.backend.schemas import (
    InspectRequest,
    InspectResponse,
    OperationGraphResponse,
)
from viewer.backend.services.inspection import InspectionService

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
async def inspect(
    request: InspectRequest,
    service: Annotated[InspectionService, Depends(get_inspection_service)],
) -> InspectResponse:
    return InspectResponse.model_validate(
        await run_blocking_io(
            service.inspect,
            model=request.model,
            preset=request.preset,
            overrides=request.overrides,
            dataset=request.dataset,
        )
    )


@router.post(
    "/inspect/operation-graph",
    response_model=OperationGraphResponse,
    summary="Inspect model preset operations",
    response_description=(
        "Torch-export operation graph for the requested model preset, or an "
        "unsupported response with warnings when tracing cannot be produced."
    ),
)
async def inspect_operation_graph(
    request: InspectRequest,
    service: Annotated[InspectionService, Depends(get_inspection_service)],
) -> OperationGraphResponse:
    return OperationGraphResponse.model_validate(
        await run_blocking_io(
            service.inspect_operation_graph,
            model=request.model,
            preset=request.preset,
            overrides=request.overrides,
            dataset=request.dataset,
        )
    )
