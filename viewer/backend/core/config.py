"""Runtime configuration for the Viewer backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from viewer.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
)

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

DEFAULT_SNAPSHOTS_ROOT = str(Path(__file__).resolve().parents[2] / "snapshots")


class ViewerApiSettings(BaseSettings):
    cors_origins: list[str] = Field(
        default_factory=lambda: LOCAL_FRONTEND_ORIGINS.copy()
    )
    logs_root: str = "logs"
    snapshots_root: str = DEFAULT_SNAPSHOTS_ROOT
    auth_mode: Literal["none", "bearer"] = "none"
    token: str | None = Field(default=None, repr=False)
    allow_unsafe_local_mutations: bool = False
    allow_log_imports: bool | None = None
    max_upload_size: int | None = Field(default=None, ge=1)
    max_log_archive_extracted_size: int = Field(
        default=DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
        ge=1,
    )
    training_cancellation_mode: Literal["strict-cgroup", "process-group"] = (
        "strict-cgroup"
    )

    model_config = SettingsConfigDict(env_prefix="VIEWER_API_")

    @model_validator(mode="after")
    def require_token_for_bearer_mode(self) -> Self:
        if self.auth_mode == "bearer" and (
            self.token is None or not self.token.strip()
        ):
            raise ValueError(
                "VIEWER_API_TOKEN must be non-empty when "
                "VIEWER_API_AUTH_MODE=bearer"
            )
        return self

    @property
    def log_imports_enabled(self) -> bool:
        if self.allow_log_imports is not None:
            return self.allow_log_imports
        return self.allow_unsafe_local_mutations or self.auth_mode == "none"


@lru_cache(maxsize=1)
def get_viewer_api_settings() -> ViewerApiSettings:
    return ViewerApiSettings()


__all__ = ["LOCAL_FRONTEND_ORIGINS", "ViewerApiSettings", "get_viewer_api_settings"]
