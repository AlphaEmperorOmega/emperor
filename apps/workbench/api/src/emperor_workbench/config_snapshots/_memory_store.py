from __future__ import annotations

from threading import RLock

from emperor_workbench.config_snapshots._errors import (
    ConfigSnapshotConflictError,
    ConfigSnapshotConflictReason,
)
from emperor_workbench.config_snapshots._records import (
    ConfigSnapshotRecord,
    snapshot_sort_key,
)
from emperor_workbench.config_snapshots._store import (
    require_no_snapshot_conflict,
    require_same_snapshot_identity,
)


class InMemoryConfigSnapshotStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, ConfigSnapshotRecord] = {}
        self._lock = RLock()

    @property
    def snapshots(self) -> dict[str, ConfigSnapshotRecord]:
        with self._lock:
            return dict(self._snapshots)

    def create(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord:
        with self._lock:
            require_no_snapshot_conflict(snapshot, self._records())
            self._snapshots[snapshot.id] = snapshot
            return snapshot

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        with self._lock:
            require_same_snapshot_identity(current, replacement)
            observed = self._snapshots.get(current.id)
            if observed is None:
                return None
            if observed != current:
                raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.STALE)
            require_no_snapshot_conflict(
                replacement,
                self._records(),
                exclude_snapshot_id=current.id,
            )
            self._snapshots[current.id] = replacement
            return replacement

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        with self._lock:
            snapshots = [
                snapshot
                for snapshot in self._snapshots.values()
                if snapshot.model == model
            ]
            return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def list_all(self) -> list[ConfigSnapshotRecord]:
        with self._lock:
            return list(self._records())

    def delete(self, snapshot_id: str) -> bool:
        with self._lock:
            return self._snapshots.pop(snapshot_id, None) is not None

    def _records(self) -> tuple[ConfigSnapshotRecord, ...]:
        return tuple(sorted(self._snapshots.values(), key=snapshot_sort_key))


__all__ = ["InMemoryConfigSnapshotStore"]
