"""Config snapshot library endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    declare_http_operation,
)
from workbench.backend.blocking import run_blocking_io
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.dependencies import (
    get_config_snapshot_service,
    get_workbench_settings,
)
from workbench.backend.model_identity import require_model_id
from workbench.backend.schemas import (
    ConfigSnapshotCreateRequest,
    ConfigSnapshotLibraryResponse,
    ConfigSnapshotResponse,
    ConfigSnapshotsResponse,
    ConfigSnapshotUpdateRequest,
)
from workbench.backend.services.config_snapshots import ConfigSnapshotService

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
    modelType: str = Query(...),
    model: str = Query(...),
) -> ConfigSnapshotsResponse:
    model_id = require_model_id(modelType, model)
    return ConfigSnapshotsResponse.model_validate(
        {
            "modelType": modelType,
            "model": model,
            "snapshots": await run_blocking_io(service.list_snapshots, model_id),
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
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def create_config_snapshot(
    request: ConfigSnapshotCreateRequest,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> ConfigSnapshotResponse:
    model_id = require_model_id(request.modelType, request.model)
    return ConfigSnapshotResponse.model_validate(
        await run_blocking_io(
            service.create_snapshot,
            model=model_id,
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
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def update_config_snapshot(
    snapshot_id: str,
    request: ConfigSnapshotUpdateRequest,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
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
@declare_http_operation(HttpOperationPolicy.LOCAL_MUTATION)
async def delete_config_snapshot(
    snapshot_id: str,
    service: Annotated[ConfigSnapshotService, Depends(get_config_snapshot_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> ConfigSnapshotsResponse:
    return ConfigSnapshotsResponse.model_validate(
        await run_blocking_io(service.delete_snapshot, snapshot_id)
    )
