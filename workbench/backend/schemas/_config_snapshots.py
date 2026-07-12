"""Config snapshot schemas."""

from __future__ import annotations

from pydantic import Field

from workbench.backend.schemas._base import (
    ApiResponseModel,
    BoundedConfigString,
    BoundedIdentifier,
    ConfigKey,
)


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
    modelType: BoundedIdentifier
    model: BoundedIdentifier
    preset: BoundedIdentifier
    name: BoundedIdentifier = ""
    overrides: dict[ConfigKey, BoundedConfigString] = Field(
        default_factory=dict,
        max_length=512,
    )


class ConfigSnapshotRenameRequest(ApiResponseModel):
    name: BoundedIdentifier


class ConfigSnapshotUpdateRequest(ApiResponseModel):
    name: BoundedIdentifier | None = None
    overrides: dict[ConfigKey, BoundedConfigString] | None = Field(
        default=None,
        max_length=512,
    )
