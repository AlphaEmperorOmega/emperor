"""Feature capability endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.dependencies import get_workbench_settings
from workbench.backend.schemas import CapabilitiesResponse
from workbench.backend.training_cgroups import requested_cancellation_capability

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
    return CapabilitiesResponse(
        authMode=settings.auth_mode,
        trainingEnabled=settings.allow_unsafe_local_mutations,
        trainingCancellationCapability=requested_cancellation_capability(
            settings.training_cancellation_mode
        ),
        logDeletionEnabled=settings.allow_unsafe_local_mutations,
        configSnapshotsEnabled=settings.allow_unsafe_local_mutations,
        uploadsEnabled=settings.log_imports_enabled,
        maxUploadSize=settings.effective_max_upload_size,
    )
