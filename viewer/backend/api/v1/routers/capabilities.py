"""Feature capability endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from viewer.backend.core.config import ViewerApiSettings
from viewer.backend.dependencies import get_viewer_settings
from viewer.backend.schemas import CapabilitiesResponse

router = APIRouter(
    tags=["capabilities"],
)


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Get API capabilities",
    response_description="Viewer API feature availability.",
)
async def capabilities(
    settings: Annotated[ViewerApiSettings, Depends(get_viewer_settings)],
) -> CapabilitiesResponse:
    return CapabilitiesResponse(authMode=settings.auth_mode)
