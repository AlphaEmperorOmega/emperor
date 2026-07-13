from __future__ import annotations

from workbench.backend.schemas._base import ApiResponseModel


class HealthResponse(ApiResponseModel):
    status: str
