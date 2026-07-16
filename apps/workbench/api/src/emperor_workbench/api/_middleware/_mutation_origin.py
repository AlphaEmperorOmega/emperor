from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from emperor_workbench.api._middleware._trusted_host import host_is_trusted
from emperor_workbench.api._mutations import HttpOperationCatalog
from emperor_workbench.api._security import (
    MUTATION_HEADER_NAME,
    MUTATION_HEADER_VALUE,
    MUTATION_PROOF_REQUIRED_DETAIL,
    UNTRUSTED_MUTATION_ORIGIN_DETAIL,
)
from emperor_workbench.settings import WorkbenchApiSettings


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
    if origin == _request_origin(scope, headers) and host_is_trusted(
        headers,
        settings,
    ):
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


__all__ = ["MutationProtectionMiddleware"]
