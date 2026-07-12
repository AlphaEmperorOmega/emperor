"""Feature capability endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    operation_policy_enabled,
)
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.dependencies import (
    get_training_job_service,
    get_workbench_settings,
)
from workbench.backend.schemas import CapabilitiesResponse
from workbench.backend.training_jobs import TrainingJobService

router = APIRouter(
    tags=["capabilities"],
)


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
    return CapabilitiesResponse(
        authMode=settings.auth_mode,
        trainingEnabled=local_mutations_enabled,
        trainingCancellationCapability=training_jobs.cancellation_capability(),
        logDeletionEnabled=local_mutations_enabled,
        configSnapshotsEnabled=local_mutations_enabled,
        uploadsEnabled=log_imports_enabled,
        maxUploadSize=settings.effective_max_upload_size,
        maxActiveTrainingJobs=settings.max_active_training_jobs,
        trainingJobMemoryLimitBytes=settings.training_job_memory_limit_bytes,
        trainingJobCpuLimit=settings.training_job_cpu_limit,
        trainingJobProcessLimit=settings.training_job_process_limit,
    )
