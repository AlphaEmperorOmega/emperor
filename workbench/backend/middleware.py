"""FastAPI middleware registration for the Workbench backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from workbench.backend.core.config import WorkbenchApiSettings


def configure_middleware(api: FastAPI, settings: WorkbenchApiSettings) -> None:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
