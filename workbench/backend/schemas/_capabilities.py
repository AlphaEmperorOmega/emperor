"""Feature capability schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from workbench.backend.schemas._base import ApiResponseModel


class CapabilitiesResponse(ApiResponseModel):
    authMode: Literal["none", "bearer"]
    trainingEnabled: bool
    trainingCancellationCapability: Literal[
        "strict-cgroup",
        "process-group",
        "unsupported",
    ] = "unsupported"
    logDeletionEnabled: bool
    configSnapshotsEnabled: bool = True
    historicalLogsEnabled: bool = True
    liveMonitorDataEnabled: bool = True
    historicalMonitorDataEnabled: bool = True
    uploadsEnabled: bool = False
    maxUploadSize: int | None = Field(default=None, ge=0)
    maxActiveTrainingJobs: int = Field(default=2, ge=1)
    trainingJobMemoryLimitBytes: int = Field(default=16 * 1024**3, ge=1)
    trainingJobCpuLimit: int = Field(default=8, ge=1)
    trainingJobProcessLimit: int = Field(default=512, ge=1)
