from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from emperor_workbench.api._container import (
    WorkbenchContainerSlot,
    activate_container,
)
from emperor_workbench.api._errors import ApiError, known_failure_response
from emperor_workbench.api._middleware._body_limit import JsonBodyLimitMiddleware
from emperor_workbench.api._middleware._mutation_origin import (
    MutationProtectionMiddleware,
)
from emperor_workbench.api._middleware._trusted_host import (
    WorkbenchTrustedHostMiddleware,
)
from emperor_workbench.api._mutations import (
    HttpOperationCatalog,
    MutationExecutionMiddleware,
)
from emperor_workbench.failures import DomainFailure
from emperor_workbench.settings import WorkbenchApiSettings

LARGE_JSON_COMPRESSION_MINIMUM_BYTES = 64 * 1024
LARGE_JSON_COMPRESSION_LEVEL = 1


class KnownFailureTranslationMiddleware:
    """Translate known failures raised above FastAPI's exception seam."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        container_slot: WorkbenchContainerSlot,
    ) -> None:
        self.app = app
        self.container_slot = container_slot

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def observe_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            try:
                container = self.container_slot.get()
            except RuntimeError:
                await self.app(scope, receive, observe_send)
            else:
                with activate_container(container):
                    await self.app(scope, receive, observe_send)
        except (ApiError, DomainFailure) as exc:
            if response_started:
                raise
            response = known_failure_response(exc)
            await response(scope, receive, send)


def configure_middleware(
    api: FastAPI,
    settings: WorkbenchApiSettings,
    operation_catalog: HttpOperationCatalog,
    *,
    container_slot: WorkbenchContainerSlot,
) -> None:
    api.add_middleware(
        MutationExecutionMiddleware,
        settings=settings,
        operation_catalog=operation_catalog,
    )
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
        KnownFailureTranslationMiddleware,
        container_slot=container_slot,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    api.add_middleware(WorkbenchTrustedHostMiddleware, settings=settings)


__all__ = ["KnownFailureTranslationMiddleware", "configure_middleware"]
