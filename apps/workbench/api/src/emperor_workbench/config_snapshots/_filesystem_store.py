from __future__ import annotations

import time
from pathlib import Path
from threading import Lock, RLock

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
from emperor_workbench.filesystem import (
    PersistentJsonCatalog,
    read_json_object,
    require_safe_name,
    resolve_root,
    resolve_under_root,
    safe_child_path,
    write_json_atomic,
)
from emperor_workbench.model_packages import ModelPackageIdentity

SNAPSHOT_FILENAME_SUFFIX = ".json"

_FILE_STORE_LOCKS_GUARD = Lock()
_FILE_STORE_LOCKS: dict[Path, RLock] = {}


def _file_store_lock(root: Path) -> RLock:
    with _FILE_STORE_LOCKS_GUARD:
        return _FILE_STORE_LOCKS.setdefault(root, RLock())


class FileSystemConfigSnapshotStore:
    def __init__(
        self,
        root: Path,
        *,
        state_root: Path | None = None,
        reconciliation_interval_seconds: float = 30.0,
    ) -> None:
        self.root = resolve_root(Path(root))
        self._lock = _file_store_lock(self.root)
        self._reconciliation_interval_seconds = max(
            0.0,
            float(reconciliation_interval_seconds),
        )
        self._catalog = (
            PersistentJsonCatalog(
                state_root=state_root,
                name="config-snapshots",
                authority_root=self.root,
            )
            if state_root is not None
            else None
        )
        self._index: dict[str, ConfigSnapshotRecord] | None = None
        self._index_generation = 0
        self._next_reconciliation = 0.0

    def create(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord:
        with self._lock:
            snapshot_path = self._snapshot_path(snapshot.model, snapshot.id)
            self._ensure_index(force=True)
            assert self._index is not None
            require_no_snapshot_conflict(snapshot, tuple(self._index.values()))
            if snapshot_path.exists() or snapshot_path.is_symlink():
                raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.ID)
            write_json_atomic(snapshot_path, _record_to_metadata(snapshot))
            self._index[snapshot.id] = snapshot
            self._index_generation += 1
            self._publish_index()
            return snapshot

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        with self._lock:
            require_same_snapshot_identity(current, replacement)
            self._ensure_index(force=True)
            observed = self._get(current.id)
            if observed is None:
                return None
            if observed != current:
                raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.STALE)
            assert self._index is not None
            require_no_snapshot_conflict(
                replacement,
                tuple(self._index.values()),
                exclude_snapshot_id=current.id,
            )
            snapshot_path = self._snapshot_path(current.model, current.id)
            write_json_atomic(snapshot_path, _record_to_metadata(replacement))
            self._index[current.id] = replacement
            self._index_generation += 1
            self._publish_index()
            return replacement

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        with self._lock:
            return self._get(snapshot_id)

    def _get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        self._snapshot_filename(snapshot_id)
        self._ensure_index()
        assert self._index is not None
        indexed = self._index.get(snapshot_id)
        if indexed is None:
            return None
        snapshot_path = self._snapshot_path(indexed.model, indexed.id)
        observed = self._read_metadata(snapshot_path)
        if observed is None or observed.id != snapshot_id:
            self._index.pop(snapshot_id, None)
            self._index_generation += 1
            self._publish_index()
            return None
        if observed != indexed:
            self._index[snapshot_id] = observed
            self._index_generation += 1
            self._publish_index()
        return observed

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        with self._lock:
            self._model_root(model)
            self._ensure_index()
            assert self._index is not None
            snapshots = [
                snapshot for snapshot in self._index.values() if snapshot.model == model
            ]
            return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def list_all(self) -> list[ConfigSnapshotRecord]:
        with self._lock:
            self._ensure_index()
            assert self._index is not None
            return sorted(self._index.values(), key=snapshot_sort_key)

    def delete(self, snapshot_id: str) -> bool:
        with self._lock:
            self._snapshot_filename(snapshot_id)
            self._ensure_index(force=True)
            assert self._index is not None
            snapshot = self._index.get(snapshot_id)
            if snapshot is None:
                return False
            snapshot_path = self._snapshot_path(snapshot.model, snapshot.id)
            if self._resolve_existing_snapshot_path(snapshot_path) is None:
                self._index.pop(snapshot_id, None)
                self._index_generation += 1
                self._publish_index()
                return False
            snapshot_path.unlink()
            self._index.pop(snapshot_id, None)
            self._index_generation += 1
            self._publish_index()
            return True

    def _scan_all(self) -> list[ConfigSnapshotRecord]:
        if not self.root.exists():
            return []
        snapshots: list[ConfigSnapshotRecord] = []
        for snapshot_path in sorted(self._root().rglob(f"*{SNAPSHOT_FILENAME_SUFFIX}")):
            resolved_path = self._resolve_existing_snapshot_path(snapshot_path)
            if resolved_path is None:
                continue
            snapshot = self._read_metadata(resolved_path)
            if snapshot is None:
                continue
            if not self._is_canonical_snapshot_path(resolved_path, snapshot):
                continue
            snapshots.append(snapshot)
        return sorted(snapshots, key=snapshot_sort_key)

    def _snapshot_path(self, model: str, snapshot_id: str) -> Path:
        model_root = self._model_root(model)
        return resolve_under_root(
            self._root(),
            model_root / self._snapshot_filename(snapshot_id),
        )

    def _ensure_index(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self._index is None and not force:
            loaded = self._load_index()
            if loaded is not None:
                records, generation = loaded
                self._index = {record.id: record for record in records}
                self._index_generation = generation
                self._next_reconciliation = now + self._reconciliation_interval_seconds
                return
        if self._index is not None and not force and now < self._next_reconciliation:
            return
        records = self._scan_all()
        self._index = {record.id: record for record in records}
        self._index_generation += 1
        self._next_reconciliation = now + self._reconciliation_interval_seconds
        self._publish_index()

    def _load_index(
        self,
    ) -> tuple[list[ConfigSnapshotRecord], int] | None:
        if self._catalog is None:
            return None
        payload = self._catalog.load(kind="config-snapshots")
        if payload is None:
            return None
        entries = payload.get("entries")
        if not isinstance(entries, list):
            return None
        records: list[ConfigSnapshotRecord] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            try:
                record = _record_from_metadata(entry)
                path = self._snapshot_path(record.model, record.id)
            except (KeyError, TypeError, ValueError):
                return None
            observed = self._read_metadata(path)
            if observed != record:
                return None
            records.append(record)
        generation = payload["generation"]
        assert isinstance(generation, int)
        return records, generation

    def _publish_index(self) -> None:
        if self._catalog is None or self._index is None:
            return
        self._catalog.publish(
            kind="config-snapshots",
            generation=self._index_generation,
            entries=[
                _record_to_metadata(snapshot)
                for snapshot in sorted(self._index.values(), key=snapshot_sort_key)
            ],
        )

    def _read_metadata(self, snapshot_path: Path) -> ConfigSnapshotRecord | None:
        resolved_path = self._resolve_existing_snapshot_path(snapshot_path)
        if resolved_path is None:
            return None
        payload = read_json_object(resolved_path)
        if payload is None:
            return None
        try:
            snapshot = _record_from_metadata(payload)
            self._model_root(snapshot.model)
            self._snapshot_filename(snapshot.id)
            return snapshot
        except (KeyError, TypeError, ValueError):
            return None

    def _root(self) -> Path:
        return resolve_root(self.root)

    def _model_root(self, model: str) -> Path:
        return safe_child_path(self._root(), model)

    def _snapshot_filename(self, snapshot_id: str) -> str:
        safe_id = require_safe_name(snapshot_id, "config snapshot id")
        if safe_id.endswith(SNAPSHOT_FILENAME_SUFFIX):
            raise ValueError("config snapshot id must be a filename stem")
        return f"{safe_id}{SNAPSHOT_FILENAME_SUFFIX}"

    def _resolve_existing_snapshot_path(self, snapshot_path: Path) -> Path | None:
        if Path(snapshot_path).is_symlink():
            return None
        try:
            resolved = resolve_under_root(self._root(), snapshot_path)
        except ValueError:
            return None
        if not resolved.is_file():
            return None
        return resolved

    def _is_canonical_snapshot_path(
        self,
        snapshot_path: Path,
        snapshot: ConfigSnapshotRecord,
    ) -> bool:
        try:
            return snapshot_path == self._snapshot_path(snapshot.model, snapshot.id)
        except ValueError:
            return False


def _record_to_metadata(snapshot: ConfigSnapshotRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": snapshot.id,
        "preset": snapshot.preset,
        "name": snapshot.name,
        "overrides": dict(snapshot.overrides),
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
    }
    try:
        identity = ModelPackageIdentity.from_id(snapshot.model)
        payload.update(
            {
                "modelType": identity.model_type,
                "model": identity.model,
            }
        )
    except ValueError:
        payload["model"] = snapshot.model
    return payload


def _record_from_metadata(payload: dict[str, object]) -> ConfigSnapshotRecord:
    overrides = payload["overrides"]
    if not isinstance(overrides, dict):
        raise TypeError("Config snapshot overrides must be a mapping.")
    identity = ModelPackageIdentity.from_mapping(payload)
    model_id = identity.catalog_key if identity is not None else str(payload["model"])
    return ConfigSnapshotRecord(
        id=str(payload["id"]),
        model=model_id,
        preset=str(payload["preset"]),
        name=str(payload["name"]),
        overrides={str(key): str(value) for key, value in overrides.items()},
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
    )


__all__ = ["FileSystemConfigSnapshotStore", "SNAPSHOT_FILENAME_SUFFIX"]
