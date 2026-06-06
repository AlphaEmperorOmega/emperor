"""Feature capability endpoint."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, Request

from viewer.backend.core.security import require_bearer_auth
from viewer.backend.schemas import CapabilitiesResponse
from viewer.backend.settings import ViewerApiSettings

router = APIRouter(
    tags=["capabilities"],
    dependencies=[Depends(require_bearer_auth)],
)


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Get API capabilities",
    response_description="Viewer API feature availability.",
)
async def capabilities(request: Request) -> CapabilitiesResponse:
    settings = cast(ViewerApiSettings, request.app.state.settings)
    return CapabilitiesResponse(authMode=settings.auth_mode)
