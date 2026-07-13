from __future__ import annotations

from typing import Any

from workbench.backend.config_snapshots import (
    ConfigSnapshotDeletion,
    ConfigSnapshotRecord,
    ConfigSnapshotService,
)
from workbench.backend.model_identity import model_identity_payload_from_id


def config_snapshot_to_payload(
    service: ConfigSnapshotService,
    snapshot: ConfigSnapshotRecord,
) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        **model_identity_payload_from_id(snapshot.model),
        "preset": snapshot.preset,
        "name": snapshot.name,
        "overrides": dict(service.canonical_overrides(snapshot)),
        "createdAt": snapshot.created_at,
        "updatedAt": snapshot.updated_at,
    }


def config_snapshots_to_payload(
    service: ConfigSnapshotService,
    model: str,
    snapshots: tuple[ConfigSnapshotRecord, ...],
) -> dict[str, Any]:
    return {
        **model_identity_payload_from_id(model),
        "snapshots": [
            config_snapshot_to_payload(service, snapshot) for snapshot in snapshots
        ],
    }


def config_snapshot_library_to_payload(
    service: ConfigSnapshotService,
    snapshots: tuple[ConfigSnapshotRecord, ...],
) -> dict[str, Any]:
    return {
        "snapshots": [
            config_snapshot_to_payload(service, snapshot) for snapshot in snapshots
        ]
    }


def config_snapshot_deletion_to_payload(
    service: ConfigSnapshotService,
    deletion: ConfigSnapshotDeletion,
) -> dict[str, Any]:
    return config_snapshots_to_payload(
        service,
        deletion.model,
        deletion.snapshots,
    )


__all__ = [
    "config_snapshot_deletion_to_payload",
    "config_snapshot_library_to_payload",
    "config_snapshot_to_payload",
    "config_snapshots_to_payload",
]
