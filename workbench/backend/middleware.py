"""FastAPI middleware registration for the Workbench backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from workbench.backend.core.config import WorkbenchApiSettings

LARGE_JSON_COMPRESSION_MINIMUM_BYTES = 64 * 1024
LARGE_JSON_COMPRESSION_LEVEL = 1


def configure_middleware(api: FastAPI, settings: WorkbenchApiSettings) -> None:
    # Scalar responses can contain tens of thousands of JSON points. A low
    # compression level removes most transfer bytes without spending level-9
    # CPU on local interactive requests; small control responses bypass it.
    api.add_middleware(
        GZipMiddleware,
        minimum_size=LARGE_JSON_COMPRESSION_MINIMUM_BYTES,
        compresslevel=LARGE_JSON_COMPRESSION_LEVEL,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
