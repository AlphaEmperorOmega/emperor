"""Data-access adapter for config snapshots.

This repository is intentionally thin: it is an extension point between services
and the concrete ``ConfigSnapshotStore`` storage adapter.
"""

from __future__ import annotations

from workbench.backend.config_snapshots import ConfigSnapshotRecord, ConfigSnapshotStore


class ConfigSnapshotRepository:
    def __init__(self, store: ConfigSnapshotStore) -> None:
        self._store = store

    def list_snapshots(self, model: str) -> list[ConfigSnapshotRecord]:
        return self._store.list(model)

    def list_all_snapshots(self) -> list[ConfigSnapshotRecord]:
        return self._store.list_all()

    def get_snapshot(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        return self._store.get(snapshot_id)

    def save_snapshot(self, snapshot: ConfigSnapshotRecord) -> None:
        self._store.save(snapshot)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        return self._store.delete(snapshot_id)
