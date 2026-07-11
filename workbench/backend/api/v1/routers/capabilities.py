"""Feature capability endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    operation_policy_enabled,
)
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.dependencies import get_workbench_settings
from workbench.backend.schemas import CapabilitiesResponse
from workbench.backend.training_jobs.cgroups import (
    requested_cancellation_capability,
)

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
        trainingCancellationCapability=requested_cancellation_capability(
            settings.training_cancellation_mode
        ),
        logDeletionEnabled=local_mutations_enabled,
        configSnapshotsEnabled=local_mutations_enabled,
        uploadsEnabled=log_imports_enabled,
        maxUploadSize=settings.effective_max_upload_size,
    )
