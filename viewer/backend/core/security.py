"""Security dependencies for the Viewer API."""

from __future__ import annotations

import secrets
from typing import cast

from fastapi import HTTPException, Request

from viewer.backend.core.config import ViewerApiSettings

UNAUTHORIZED_DETAIL = "Missing or invalid bearer credentials"


async def require_bearer_auth(request: Request) -> None:
    """Validate the configured shared bearer token when hosted auth is enabled."""
    settings = cast(ViewerApiSettings, request.app.state.settings)
    if settings.auth_mode == "none":
        return

    authorization = request.headers.get("Authorization")
    bearer_token = _parse_bearer_token(authorization)
    configured_token = settings.token

    if (
        bearer_token is None
        or configured_token is None
        or not secrets.compare_digest(bearer_token, configured_token)
    ):
        raise HTTPException(status_code=401, detail=UNAUTHORIZED_DETAIL)


def _parse_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None

    parts = authorization.split()
    if len(parts) != 2:
        return None

    scheme, token = parts
    if scheme.lower() != "bearer" or not token:
        return None
    return token


__all__ = ["require_bearer_auth"]
