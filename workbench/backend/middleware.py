"""FastAPI middleware registration for the Workbench backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from workbench.backend.api.mutation_policy import HttpOperationCatalog
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import (
    MUTATION_HEADER_NAME,
    MUTATION_HEADER_VALUE,
    MUTATION_PROOF_REQUIRED_DETAIL,
    UNTRUSTED_MUTATION_ORIGIN_DETAIL,
)
from workbench.backend.mutation_execution import MutationExecutionMiddleware

LARGE_JSON_COMPRESSION_MINIMUM_BYTES = 64 * 1024
LARGE_JSON_COMPRESSION_LEVEL = 1


def _is_json_content_type(headers: Headers) -> bool:
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


class JsonBodyLimitMiddleware:
    """Stream JSON request bodies into a bounded replay buffer."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max(1, int(max_bytes))

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not _is_json_content_type(headers):
            await self.app(scope, receive, send)
            return

        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = 0
            if declared_bytes > self.max_bytes:
                await self._reject(scope, receive, send)
                return

        messages: list[dict[str, object]] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                messages.append(message)
                break
            body = message.get("body", b"")
            if isinstance(body, bytes):
                received_bytes += len(body)
            if received_bytes > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay() -> dict[str, object]:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)  # type: ignore[arg-type]

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(
            {"detail": (f"JSON request body exceeds the {self.max_bytes} byte limit.")},
            status_code=413,
        )
        await response(scope, receive, send)


def _authority_host(authority: str) -> str:
    value = authority.strip().casefold()
    if value.startswith("["):
        closing_bracket = value.find("]")
        return value[: closing_bracket + 1] if closing_bracket >= 0 else value
    return value.split(":", 1)[0].rstrip(".")


def _host_matches_pattern(host: str, pattern: str) -> bool:
    normalized_pattern = _authority_host(pattern)
    if normalized_pattern == "*":
        return True
    if normalized_pattern.startswith("*."):
        suffix = normalized_pattern[1:]
        return host.endswith(suffix) and host != suffix[1:]
    return host == normalized_pattern


def _host_is_trusted(headers: Headers, settings: WorkbenchApiSettings) -> bool:
    host = _authority_host(headers.get("host", ""))
    return bool(host) and any(
        _host_matches_pattern(host, pattern) for pattern in settings.trusted_hosts
    )


class WorkbenchTrustedHostMiddleware:
    """Reject unconfigured authorities before any route policy executes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: WorkbenchApiSettings,
    ) -> None:
        self.app = app
        self.settings = settings

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if not _host_is_trusted(Headers(scope=scope), self.settings):
            response = PlainTextResponse("Invalid host header", status_code=400)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


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
    if origin == _request_origin(scope, headers) and _host_is_trusted(
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


def configure_middleware(
    api: FastAPI,
    settings: WorkbenchApiSettings,
    operation_catalog: HttpOperationCatalog,
) -> None:
    api.add_middleware(
        MutationExecutionMiddleware,
        settings=settings,
        operation_catalog=operation_catalog,
    )
    # Scalar responses can contain tens of thousands of JSON points. A low
    # compression level removes most transfer bytes without spending level-9
    # CPU on local interactive requests; small control responses bypass it.
    api.add_middleware(
        GZipMiddleware,
        minimum_size=LARGE_JSON_COMPRESSION_MINIMUM_BYTES,
        compresslevel=LARGE_JSON_COMPRESSION_LEVEL,
    )
    api.add_middleware(
        JsonBodyLimitMiddleware,
        max_bytes=settings.max_json_body_bytes,
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
    # Starlette executes the most recently registered middleware first.
    api.add_middleware(WorkbenchTrustedHostMiddleware, settings=settings)
