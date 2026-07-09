"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from workbench.backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    response_description="Workbench API health status.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
