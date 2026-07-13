from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from workbench.backend.api.mutation_policy import (
    HttpOperationPolicy,
    declare_http_operation,
)
from workbench.backend.api.v1.config_snapshot_mapping import (
    config_snapshot_deletion_to_payload,
    config_snapshot_library_to_payload,
    config_snapshot_to_payload,
    config_snapshots_to_payload,
)
from workbench.backend.blocking import run_blocking_io
from workbench.backend.config_snapshots import ConfigSnapshotService
from workbench.backend.core.config import WorkbenchApiSettings
from workbench.backend.core.security import require_bearer_auth
from workbench.backend.dependencies import (
    get_config_snapshot_service,
    get_workbench_settings,
)
from workbench.backend.model_identity import require_model_id
from workbench.backend.mutation_execution import run_mutation_io
from workbench.backend.schemas import (
    ConfigSnapshotCreateRequest,
    ConfigSnapshotLibraryResponse,
    ConfigSnapshotResponse,
    ConfigSnapshotsResponse,
    ConfigSnapshotUpdateRequest,
)

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
    modelType: Annotated[str, Query()],
    model: Annotated[str, Query()],
) -> ConfigSnapshotsResponse:
    model_id = require_model_id(modelType, model)
    snapshots = await run_blocking_io(service.list_snapshots, model_id)
    return ConfigSnapshotsResponse.model_validate(
        config_snapshots_to_payload(service, model_id, snapshots)
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
    snapshots = await run_blocking_io(service.list_all_snapshots)
    return ConfigSnapshotLibraryResponse.model_validate(
        config_snapshot_library_to_payload(service, snapshots)
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
    snapshot = await run_mutation_io(
        service.create_snapshot,
        model=model_id,
        preset=request.preset,
        name=request.name,
        overrides=request.overrides,
    )
    return ConfigSnapshotResponse.model_validate(
        config_snapshot_to_payload(service, snapshot)
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
    snapshot = await run_mutation_io(
        service.update_snapshot,
        snapshot_id,
        name=request.name,
        overrides=request.overrides,
    )
    return ConfigSnapshotResponse.model_validate(
        config_snapshot_to_payload(service, snapshot)
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
    deletion = await run_mutation_io(service.delete_snapshot, snapshot_id)
    return ConfigSnapshotsResponse.model_validate(
        config_snapshot_deletion_to_payload(service, deletion)
    )
