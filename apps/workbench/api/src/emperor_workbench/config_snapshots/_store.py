from __future__ import annotations

from typing import Protocol

from emperor_workbench.config_snapshots._errors import (
    ConfigSnapshotConflictError,
    ConfigSnapshotConflictReason,
)
from emperor_workbench.config_snapshots._records import ConfigSnapshotRecord


class ConfigSnapshotStore(Protocol):
    def create(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord: ...

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None: ...

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None: ...

    def list(self, model: str) -> list[ConfigSnapshotRecord]: ...

    def list_all(self) -> list[ConfigSnapshotRecord]: ...

    def delete(self, snapshot_id: str) -> bool: ...


def require_same_snapshot_identity(
    current: ConfigSnapshotRecord,
    replacement: ConfigSnapshotRecord,
) -> None:
    if (
        replacement.id != current.id
        or replacement.model != current.model
        or replacement.preset != current.preset
        or replacement.created_at != current.created_at
    ):
        raise ValueError("config snapshot update cannot change record identity")


def require_no_snapshot_conflict(
    candidate: ConfigSnapshotRecord,
    existing_snapshots: tuple[ConfigSnapshotRecord, ...],
    *,
    exclude_snapshot_id: str | None = None,
) -> None:
    candidate_name = candidate.name.strip().casefold()
    candidate_overrides = dict(candidate.overrides)
    for existing in existing_snapshots:
        if existing.id == exclude_snapshot_id:
            continue
        if existing.id == candidate.id:
            raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.ID)
        if existing.model != candidate.model or existing.preset != candidate.preset:
            continue
        if existing.name.strip().casefold() == candidate_name:
            raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.NAME)
        if dict(existing.overrides) == candidate_overrides:
            raise ConfigSnapshotConflictError(
                ConfigSnapshotConflictReason.RUNTIME_DEFAULTS
            )


__all__ = [
    "ConfigSnapshotStore",
    "require_no_snapshot_conflict",
    "require_same_snapshot_identity",
]
