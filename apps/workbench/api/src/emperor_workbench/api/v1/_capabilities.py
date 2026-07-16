from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import Field

from emperor_workbench.api._dependencies import (
    get_training_job_service,
    get_workbench_settings,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    operation_policy_enabled,
)
from emperor_workbench.api.v1._base_contracts import ApiResponseModel
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService


class CapabilitiesResponse(ApiResponseModel):
    authMode: Literal["none", "bearer"]
    trainingEnabled: bool
    trainingCancellationCapability: Literal[
        "strict-cgroup",
        "process-group",
        "windows-job-object",
        "unsupported",
    ] = "unsupported"
    trainingResourceLimitsEnforced: bool = False
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


router = APIRouter(tags=["capabilities"])


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Get API capabilities",
    response_description="Workbench API feature availability.",
)
async def capabilities(
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
    training_jobs: Annotated[
        TrainingJobService,
        Depends(get_training_job_service),
    ],
) -> CapabilitiesResponse:
    local_mutations_enabled = operation_policy_enabled(
        HttpOperationPolicy.LOCAL_MUTATION,
        settings,
    )
    log_imports_enabled = operation_policy_enabled(
        HttpOperationPolicy.LOG_IMPORT,
        settings,
    )
    cancellation_capability = training_jobs.cancellation_capability()
    return CapabilitiesResponse(
        authMode=settings.auth_mode,
        trainingEnabled=local_mutations_enabled,
        trainingCancellationCapability=cancellation_capability,
        trainingResourceLimitsEnforced=cancellation_capability
        in {"strict-cgroup", "windows-job-object"},
        logDeletionEnabled=local_mutations_enabled,
        configSnapshotsEnabled=local_mutations_enabled,
        uploadsEnabled=log_imports_enabled,
        maxUploadSize=settings.effective_max_upload_size,
        maxActiveTrainingJobs=settings.max_active_training_jobs,
        trainingJobMemoryLimitBytes=settings.training_job_memory_limit_bytes,
        trainingJobCpuLimit=settings.training_job_cpu_limit,
        trainingJobProcessLimit=settings.training_job_process_limit,
    )


__all__ = ["CapabilitiesResponse", "router"]
