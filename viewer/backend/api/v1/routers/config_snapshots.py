"""Config snapshot library endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from viewer.backend.blocking import run_blocking_io
from viewer.backend.core.security import require_bearer_auth
from viewer.backend.dependencies import get_config_snapshot_service
from viewer.backend.schemas import (
    ConfigSnapshotCreateRequest,
    ConfigSnapshotLibraryResponse,
    ConfigSnapshotResponse,
    ConfigSnapshotsResponse,
    ConfigSnapshotUpdateRequest,
)
from viewer.backend.services.config_snapshots import ConfigSnapshotService

router = APIRouter(
    prefix="/config-snapshots",
    tags=["config-snapshots"],
    dependencies=[Depends(require_bearer_auth)],
)


@router.get(
    "",
    response_model=ConfigSnapshotsResponse,
    summary="List config snapshots for a model",
    response_description="Stored config snapshots for the requested model.",
)
async def list_config_snapshots(
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
    model: str = Query(...),
) -> ConfigSnapshotsResponse:
    return ConfigSnapshotsResponse.model_validate(
        {
            "model": model,
            "snapshots": await run_blocking_io(service.list_snapshots, model),
        }
    )


@router.get(
    "/library",
    response_model=ConfigSnapshotLibraryResponse,
    summary="List all config snapshots",
    response_description="Stored config snapshots across all models.",
)
async def list_config_snapshot_library(
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
) -> ConfigSnapshotLibraryResponse:
    return ConfigSnapshotLibraryResponse.model_validate(
        {"snapshots": await run_blocking_io(service.list_all_snapshots)}
    )


@router.post(
    "",
    response_model=ConfigSnapshotResponse,
    summary="Create a config snapshot",
    response_description="The created config snapshot.",
)
async def create_config_snapshot(
    request: ConfigSnapshotCreateRequest,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
) -> ConfigSnapshotResponse:
    return ConfigSnapshotResponse.model_validate(
        await run_blocking_io(
            service.create_snapshot,
            model=request.model,
            preset=request.preset,
            name=request.name,
            overrides=request.overrides,
        )
    )


@router.patch(
    "/{snapshot_id}",
    response_model=ConfigSnapshotResponse,
    summary="Update a config snapshot",
    response_description="The updated config snapshot.",
)
async def update_config_snapshot(
    snapshot_id: str,
    request: ConfigSnapshotUpdateRequest,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
) -> ConfigSnapshotResponse:
    return ConfigSnapshotResponse.model_validate(
        await run_blocking_io(
            service.update_snapshot,
            snapshot_id,
            name=request.name,
            overrides=request.overrides,
        )
    )


@router.delete(
    "/{snapshot_id}",
    response_model=ConfigSnapshotsResponse,
    summary="Delete a config snapshot",
    response_description="Remaining config snapshots for the model.",
)
async def delete_config_snapshot(
    snapshot_id: str,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
) -> ConfigSnapshotsResponse:
    return ConfigSnapshotsResponse.model_validate(
        await run_blocking_io(service.delete_snapshot, snapshot_id)
    )
