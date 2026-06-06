"""Runtime settings for the viewer API (CORS origins, logs root)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
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
    snapshots_root: str = "snapshots"
    auth_mode: Literal["none", "bearer"] = "none"
    token: str | None = Field(default=None, repr=False)

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
