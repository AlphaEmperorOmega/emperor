"""FastAPI middleware registration for the Workbench backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from workbench.backend.api.mutation_policy import HttpOperationCatalog
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import (
    MUTATION_HEADER_NAME,
    MUTATION_HEADER_VALUE,
    MUTATION_PROOF_REQUIRED_DETAIL,
    UNTRUSTED_MUTATION_ORIGIN_DETAIL,
)

LARGE_JSON_COMPRESSION_MINIMUM_BYTES = 64 * 1024
LARGE_JSON_COMPRESSION_LEVEL = 1


def _request_origin(scope: Scope, headers: Headers) -> str | None:
    host = headers.get("host")
    if not host:
        return None
    return f"{scope.get('scheme', 'http')}://{host}"


def _origin_is_trusted(
    scope: Scope,
    headers: Headers,
    settings: WorkbenchApiSettings,
) -> bool:
    origin = headers.get("origin")
    if origin is None:
        return True
    if origin == _request_origin(scope, headers):
        return True
    return origin in settings.cors_origins


class MutationProtectionMiddleware:
    """Reject forged browser mutations before request bodies are consumed."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: WorkbenchApiSettings,
        operation_catalog: HttpOperationCatalog,
    ) -> None:
        self.app = app
        self.settings = settings
        self.operation_catalog = operation_catalog

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or self.operation_catalog.mutation_for_scope(scope) is None
        ):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin")
        fetch_site = headers.get("sec-fetch-site", "").casefold()
        origin_is_trusted = _origin_is_trusted(scope, headers, self.settings)
        cross_site_origin_is_configured = (
            fetch_site == "cross-site"
            and origin is not None
            and origin in self.settings.cors_origins
        )
        if not origin_is_trusted or (
            fetch_site == "cross-site" and not cross_site_origin_is_configured
        ):
            await self._reject(
                scope,
                receive,
                send,
                detail=UNTRUSTED_MUTATION_ORIGIN_DETAIL,
            )
            return

        if (
            self.settings.auth_mode == "none"
            and headers.get(MUTATION_HEADER_NAME) != MUTATION_HEADER_VALUE
        ):
            await self._reject(
                scope,
                receive,
                send,
                detail=MUTATION_PROOF_REQUIRED_DETAIL,
            )
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        detail: str,
    ) -> None:
        response = JSONResponse({"detail": detail}, status_code=403)
        await response(scope, receive, send)


def configure_middleware(
    api: FastAPI,
    settings: WorkbenchApiSettings,
    operation_catalog: HttpOperationCatalog,
) -> None:
    # Scalar responses can contain tens of thousands of JSON points. A low
    # compression level removes most transfer bytes without spending level-9
    # CPU on local interactive requests; small control responses bypass it.
    api.add_middleware(
        GZipMiddleware,
        minimum_size=LARGE_JSON_COMPRESSION_MINIMUM_BYTES,
        compresslevel=LARGE_JSON_COMPRESSION_LEVEL,
    )
    api.add_middleware(
        MutationProtectionMiddleware,
        settings=settings,
        operation_catalog=operation_catalog,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
