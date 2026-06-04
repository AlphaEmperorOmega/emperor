"""FastAPI middleware registration for the Viewer backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from viewer.backend.core.config import ViewerApiSettings


def configure_middleware(api: FastAPI, settings: ViewerApiSettings) -> None:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
