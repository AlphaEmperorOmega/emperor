"""Model inspection endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from viewer.backend.dependencies import get_inspection_service
from viewer.backend.schemas import InspectRequest, InspectResponse
from viewer.backend.services.inspection import InspectionService

router = APIRouter(tags=["inspection"])


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
        service.inspect(
            model=request.model,
            preset=request.preset,
            overrides=request.overrides,
            dataset=request.dataset,
        )
    )
