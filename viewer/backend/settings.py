"""Runtime settings for the viewer API (CORS origins, logs root)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:9000",
    "http://127.0.0.1:9000",
    "http://0.0.0.0:9000",
    "http://localhost:9001",
    "http://127.0.0.1:9001",
    "http://0.0.0.0:9001",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
]


class ViewerApiSettings(BaseSettings):
    cors_origins: list[str] = Field(default_factory=lambda: LOCAL_FRONTEND_ORIGINS.copy())
    logs_root: str = "logs"

    model_config = SettingsConfigDict(env_prefix="VIEWER_API_")
