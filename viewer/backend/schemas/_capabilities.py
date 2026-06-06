"""Feature capability schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel


class DataSourceCapabilityPlaceholder(ApiResponseModel):
    """Reserved shape for future server-owned data source descriptors."""


class CapabilitiesResponse(ApiResponseModel):
    authMode: Literal["none", "bearer"]
    trainingEnabled: bool = True
    logDeletionEnabled: bool = True
    historicalLogsEnabled: bool = True
    liveMonitorDataEnabled: bool = True
    historicalMonitorDataEnabled: bool = True
    uploadsEnabled: bool = False
    maxUploadSize: int | None = Field(default=None, ge=0)
    dataSourcesEnabled: bool = False
    dataSources: list[DataSourceCapabilityPlaceholder] = Field(default_factory=list)
