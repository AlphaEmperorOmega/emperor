from __future__ import annotations

from emperor_workbench.api.v1.config_snapshots._contracts import (
    ConfigSnapshotLibraryResponse,
    ConfigSnapshotResponse,
    ConfigSnapshotsResponse,
)
from emperor_workbench.config_snapshots import (
    ConfigSnapshotDeletion,
    ConfigSnapshotRecord,
    ConfigSnapshotService,
)
from emperor_workbench.model_packages import ModelPackageIdentity


def _model_identity(model_id: str) -> ModelPackageIdentity:
    return ModelPackageIdentity.from_id(model_id)


def config_snapshot_response(
    service: ConfigSnapshotService,
    snapshot: ConfigSnapshotRecord,
) -> ConfigSnapshotResponse:
    identity = _model_identity(snapshot.model)
    return ConfigSnapshotResponse(
        id=snapshot.id,
        modelType=identity.model_type,
        model=identity.model,
        preset=snapshot.preset,
        name=snapshot.name,
        overrides=dict(service.canonical_overrides(snapshot)),
        createdAt=snapshot.created_at,
        updatedAt=snapshot.updated_at,
    )


def config_snapshots_response(
    service: ConfigSnapshotService,
    model: str,
    snapshots: tuple[ConfigSnapshotRecord, ...],
) -> ConfigSnapshotsResponse:
    identity = _model_identity(model)
    return ConfigSnapshotsResponse(
        modelType=identity.model_type,
        model=identity.model,
        snapshots=[
            config_snapshot_response(service, snapshot) for snapshot in snapshots
        ],
    )


def config_snapshot_library_response(
    service: ConfigSnapshotService,
    snapshots: tuple[ConfigSnapshotRecord, ...],
) -> ConfigSnapshotLibraryResponse:
    return ConfigSnapshotLibraryResponse(
        snapshots=[
            config_snapshot_response(service, snapshot) for snapshot in snapshots
        ]
    )


def config_snapshot_deletion_response(
    service: ConfigSnapshotService,
    deletion: ConfigSnapshotDeletion,
) -> ConfigSnapshotsResponse:
    return config_snapshots_response(
        service,
        deletion.model,
        deletion.snapshots,
    )


__all__ = [
    "config_snapshot_deletion_response",
    "config_snapshot_library_response",
    "config_snapshot_response",
    "config_snapshots_response",
]
