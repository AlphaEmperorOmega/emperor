from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from emperor_workbench.api._blocking import run_blocking_io
from emperor_workbench.api._dependencies import (
    get_config_snapshot_service,
    get_project_adapter_client,
    get_workbench_settings,
)
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
    deterministic_mutation_resource_id,
    run_mutation_io,
)
from emperor_workbench.api._security import require_bearer_auth
from emperor_workbench.api.v1.config_snapshots._contracts import (
    ConfigSnapshotCreateRequest,
    ConfigSnapshotLibraryResponse,
    ConfigSnapshotResponse,
    ConfigSnapshotsResponse,
    ConfigSnapshotUpdateRequest,
)
from emperor_workbench.api.v1.config_snapshots._mapping import (
    config_snapshot_deletion_response,
    config_snapshot_library_response,
    config_snapshot_response,
    config_snapshots_response,
)
from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings

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
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> ConfigSnapshotsResponse:
    model_id = await run_blocking_io(
        ModelPackageCatalog(project_adapter).require_id,
        modelType,
        model,
    )
    snapshots = await run_blocking_io(service.list_snapshots, model_id)
    return await run_blocking_io(
        config_snapshots_response,
        service,
        model_id,
        snapshots,
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
    return await run_blocking_io(
        config_snapshot_library_response,
        service,
        snapshots,
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
    project_adapter: Annotated[
        ProjectAdapterClient,
        Depends(get_project_adapter_client),
    ],
) -> ConfigSnapshotResponse:
    model_id = await run_blocking_io(
        ModelPackageCatalog(project_adapter).require_id,
        request.modelType,
        request.model,
    )
    snapshot = await run_mutation_io(
        service.create_snapshot,
        model=model_id,
        preset=request.preset,
        name=request.name,
        overrides=request.overrides,
        snapshot_id=deterministic_mutation_resource_id("config-snapshot"),
    )
    return await run_blocking_io(config_snapshot_response, service, snapshot)


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
    return await run_blocking_io(config_snapshot_response, service, snapshot)


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
    return await run_blocking_io(
        config_snapshot_deletion_response,
        service,
        deletion,
    )


__all__ = ["router"]
