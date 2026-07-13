from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from workbench.backend.core.limits import (
    DEFAULT_LOG_ARCHIVE_UPLOAD_CONCURRENCY,
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
    DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
    DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE,
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
DEFAULT_TRUSTED_HOSTS = ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"]

DEFAULT_SNAPSHOTS_ROOT = str(Path(__file__).resolve().parents[2] / "snapshots")
DEFAULT_STATE_ROOT = str(Path(__file__).resolve().parents[2] / ".runtime" / "state")


class WorkbenchApiSettings(BaseSettings):
    cors_origins: list[str] = Field(
        default_factory=lambda: LOCAL_FRONTEND_ORIGINS.copy()
    )
    trusted_hosts: list[str] = Field(
        default_factory=lambda: DEFAULT_TRUSTED_HOSTS.copy(),
        min_length=1,
    )
    logs_root: str = "logs"
    snapshots_root: str = DEFAULT_SNAPSHOTS_ROOT
    state_root: str = DEFAULT_STATE_ROOT
    auth_mode: Literal["none", "bearer"] = "none"
    token: str | None = Field(default=None, repr=False)
    allow_unsafe_local_mutations: bool = False
    allow_log_imports: bool | None = None
    max_upload_size: int | None = Field(default=None, ge=1)
    max_log_archive_extracted_size: int | None = Field(default=None, ge=1)
    max_log_archive_member_count: int = Field(
        default=DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
        ge=1,
    )
    max_log_archive_path_bytes: int = Field(
        default=DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
        ge=1,
    )
    log_archive_upload_concurrency: int = Field(
        default=DEFAULT_LOG_ARCHIVE_UPLOAD_CONCURRENCY,
        ge=1,
    )
    training_cancellation_mode: Literal[
        "auto",
        "strict-cgroup",
        "process-group",
        "windows-job-object",
    ] = "auto"
    inspection_memory_limit_bytes: int = Field(default=4 * 1024**3, ge=1)
    inspection_cpu_limit: int = Field(default=4, ge=1)
    inspection_timeout_seconds: float = Field(default=60.0, gt=0)
    max_json_body_bytes: int = Field(default=1024**2, ge=1)
    tensorboard_request_work_bytes: int = Field(default=64 * 1024**2, ge=1)
    tensorboard_cache_bytes: int = Field(default=128 * 1024**2, ge=1)
    max_progress_record_bytes: int = Field(default=1024**2, ge=1)
    max_active_training_jobs: int = Field(default=2, ge=1)
    training_job_memory_limit_bytes: int = Field(default=16 * 1024**3, ge=1)
    training_job_cpu_limit: int = Field(default=8, ge=1)
    training_job_process_limit: int = Field(default=512, ge=1)

    model_config = SettingsConfigDict(env_prefix="WORKBENCH_API_")

    @model_validator(mode="after")
    def require_token_for_bearer_mode(self) -> Self:
        if self.auth_mode == "bearer" and (
            self.token is None or not self.token.strip()
        ):
            raise ValueError(
                "WORKBENCH_API_TOKEN must be non-empty when "
                "WORKBENCH_API_AUTH_MODE=bearer"
            )
        return self

    @property
    def log_imports_enabled(self) -> bool:
        return self.allow_log_imports is True

    @property
    def effective_max_upload_size(self) -> int:
        if self.max_upload_size is not None:
            return self.max_upload_size
        return DEFAULT_MAX_LOG_ARCHIVE_UPLOAD_SIZE

    @property
    def effective_max_log_archive_extracted_size(self) -> int:
        if self.max_log_archive_extracted_size is not None:
            return self.max_log_archive_extracted_size
        return DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE


@lru_cache(maxsize=1)
def get_workbench_api_settings() -> WorkbenchApiSettings:
    return WorkbenchApiSettings()


__all__ = [
    "DEFAULT_STATE_ROOT",
    "DEFAULT_TRUSTED_HOSTS",
    "LOCAL_FRONTEND_ORIGINS",
    "WorkbenchApiSettings",
    "get_workbench_api_settings",
]
