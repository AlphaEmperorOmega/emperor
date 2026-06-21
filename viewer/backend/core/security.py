"""Security dependencies for the Viewer API."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from viewer.backend.core.config import ViewerApiSettings
from viewer.backend.dependencies import get_viewer_settings

UNAUTHORIZED_DETAIL = "Missing or invalid bearer credentials"
WWW_AUTHENTICATE_HEADER = "Bearer"
LOCAL_MUTATION_DISABLED_DETAIL = "Local mutation endpoints are disabled"

bearer_scheme = HTTPBearer(auto_error=False)


async def require_bearer_auth(
    settings: Annotated[ViewerApiSettings, Depends(get_viewer_settings)],
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> None:
    """Validate the configured shared bearer token when hosted auth is enabled."""
    if settings.auth_mode == "none":
        return

    configured_token = settings.token

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not credentials.credentials
        or configured_token is None
        or not secrets.compare_digest(credentials.credentials, configured_token)
    ):
        raise HTTPException(
            status_code=401,
            detail=UNAUTHORIZED_DETAIL,
            headers={"WWW-Authenticate": WWW_AUTHENTICATE_HEADER},
        )


def require_local_mutations_allowed(settings: ViewerApiSettings) -> None:
    """Fail closed for endpoints that mutate local files or processes."""

    if settings.allow_unsafe_local_mutations:
        return
    raise HTTPException(
        status_code=403,
        detail=LOCAL_MUTATION_DISABLED_DETAIL,
    )


__all__ = [
    "WWW_AUTHENTICATE_HEADER",
    "bearer_scheme",
    "require_local_mutations_allowed",
    "require_bearer_auth",
]
