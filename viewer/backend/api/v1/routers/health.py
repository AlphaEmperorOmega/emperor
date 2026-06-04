"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from viewer.backend.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    response_description="Viewer API health status.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
