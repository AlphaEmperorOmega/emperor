"""Config snapshot schemas."""

from __future__ import annotations

from pydantic import Field

from workbench.backend.schemas._base import ApiResponseModel


class ConfigSnapshotResponse(ApiResponseModel):
    id: str
    modelType: str
    model: str
    preset: str
    name: str
    overrides: dict[str, str]
    createdAt: str
    updatedAt: str


class ConfigSnapshotsResponse(ApiResponseModel):
    modelType: str
    model: str
    snapshots: list[ConfigSnapshotResponse]


class ConfigSnapshotLibraryResponse(ApiResponseModel):
    snapshots: list[ConfigSnapshotResponse]


class ConfigSnapshotCreateRequest(ApiResponseModel):
    modelType: str
    model: str
    preset: str
    name: str = ""
    overrides: dict[str, str] = Field(default_factory=dict)


class ConfigSnapshotRenameRequest(ApiResponseModel):
    name: str


class ConfigSnapshotUpdateRequest(ApiResponseModel):
    name: str | None = None
    overrides: dict[str, str] | None = None
