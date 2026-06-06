"""Health-check schemas."""

from __future__ import annotations

from viewer.backend.schemas._base import ApiResponseModel


class HealthResponse(ApiResponseModel):
    status: str
