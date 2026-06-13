"""Config snapshot schemas."""

from __future__ import annotations

from pydantic import Field

from viewer.backend.schemas._base import ApiResponseModel


class ConfigSnapshotResponse(ApiResponseModel):
    id: str
    model: str
    preset: str
    name: str
    overrides: dict[str, str]
    createdAt: str
    updatedAt: str


class ConfigSnapshotsResponse(ApiResponseModel):
    model: str
    snapshots: list[ConfigSnapshotResponse]


class ConfigSnapshotLibraryResponse(ApiResponseModel):
    snapshots: list[ConfigSnapshotResponse]


class ConfigSnapshotCreateRequest(ApiResponseModel):
    model: str
    preset: str
    name: str = ""
    overrides: dict[str, str] = Field(default_factory=dict)


class ConfigSnapshotRenameRequest(ApiResponseModel):
    name: str
