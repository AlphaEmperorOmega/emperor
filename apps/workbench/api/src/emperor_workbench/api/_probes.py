from __future__ import annotations

from fastapi import APIRouter

from emperor_workbench.api.v1._base_contracts import ApiResponseModel


class HealthResponse(ApiResponseModel):
    status: str


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    response_description="Workbench API health status.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


__all__ = ["HealthResponse", "router"]
