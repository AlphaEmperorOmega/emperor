from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from emperor_workbench.api._dependencies import get_workbench_settings
from emperor_workbench.settings import WorkbenchApiSettings

UNAUTHORIZED_DETAIL = "Missing or invalid bearer credentials"
WWW_AUTHENTICATE_HEADER = "Bearer"
LOCAL_MUTATION_DISABLED_DETAIL = "Local mutation endpoints are disabled"
MUTATION_HEADER_NAME = "X-Workbench-Mutation"
MUTATION_HEADER_VALUE = "true"
MUTATION_PROOF_REQUIRED_DETAIL = "Mutation request proof is missing or invalid"
UNTRUSTED_MUTATION_ORIGIN_DETAIL = "Mutation request origin is not trusted"

bearer_scheme = HTTPBearer(auto_error=False)


async def require_bearer_auth(
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
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


__all__ = [
    "MUTATION_HEADER_NAME",
    "MUTATION_HEADER_VALUE",
    "MUTATION_PROOF_REQUIRED_DETAIL",
    "UNTRUSTED_MUTATION_ORIGIN_DETAIL",
    "WWW_AUTHENTICATE_HEADER",
    "bearer_scheme",
    "require_bearer_auth",
]
