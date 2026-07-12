"""Config snapshot record storage interfaces and local adapters.

A config snapshot captures a named set of config overrides for a given
``model + preset`` so it can be reused and trained later. The Workbench backend has
no database, so persistence mirrors :mod:`workbench.backend.training_jobs.store`:
records are
serialized to JSON on disk, one file per snapshot under
``<root>/<model>/<id>.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock, RLock
from types import MappingProxyType
from typing import Protocol

from emperor.model_packages import (
    model_id_from_payload,
    model_identity_payload_from_id,
)

from workbench.backend.storage.local_files import (
    read_json_object,
    require_safe_name,
    resolve_root,
    resolve_under_root,
    safe_child_path,
    write_json_atomic,
)

SNAPSHOT_FILENAME_SUFFIX = ".json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ConfigSnapshotRecord:
    id: str
    model: str
    preset: str
    name: str
    overrides: Mapping[str, str]
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(dict(self.overrides)),
        )


class ConfigSnapshotConflictReason(StrEnum):
    ID = "id"
    NAME = "name"
    RUNTIME_DEFAULTS = "runtime-defaults"
    STALE = "stale"


class ConfigSnapshotConflictError(Exception):
    def __init__(self, reason: ConfigSnapshotConflictReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


class ConfigSnapshotStore(Protocol):
    def create(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord:
        ...

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        ...

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        ...

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        ...

    def list_all(self) -> list[ConfigSnapshotRecord]:
        ...

    def delete(self, snapshot_id: str) -> bool:
        ...


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
            _require_no_snapshot_conflict(snapshot, self._records())
            self._snapshots[snapshot.id] = snapshot
            return snapshot

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        with self._lock:
            _require_same_snapshot_identity(current, replacement)
            observed = self._snapshots.get(current.id)
            if observed is None:
                return None
            if observed != current:
                raise ConfigSnapshotConflictError(
                    ConfigSnapshotConflictReason.STALE
                )
            _require_no_snapshot_conflict(
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
        return tuple(sorted(self._snapshots.values(), key=_snapshot_sort_key))


_FILE_STORE_LOCKS_GUARD = Lock()
_FILE_STORE_LOCKS: dict[Path, RLock] = {}


def _file_store_lock(root: Path) -> RLock:
    with _FILE_STORE_LOCKS_GUARD:
        return _FILE_STORE_LOCKS.setdefault(root, RLock())


class FileSystemConfigSnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = resolve_root(Path(root))
        self._lock = _file_store_lock(self.root)

    def create(self, snapshot: ConfigSnapshotRecord) -> ConfigSnapshotRecord:
        with self._lock:
            snapshot_path = self._snapshot_path(snapshot.model, snapshot.id)
            _require_no_snapshot_conflict(snapshot, tuple(self._list_all()))
            if snapshot_path.exists() or snapshot_path.is_symlink():
                raise ConfigSnapshotConflictError(
                    ConfigSnapshotConflictReason.ID
                )
            write_json_atomic(snapshot_path, _record_to_metadata(snapshot))
            return snapshot

    def update(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        with self._lock:
            _require_same_snapshot_identity(current, replacement)
            observed = self._get(current.id)
            if observed is None:
                return None
            if observed != current:
                raise ConfigSnapshotConflictError(
                    ConfigSnapshotConflictReason.STALE
                )
            _require_no_snapshot_conflict(
                replacement,
                tuple(self._list_all()),
                exclude_snapshot_id=current.id,
            )
            snapshot_path = self._snapshot_path(current.model, current.id)
            write_json_atomic(snapshot_path, _record_to_metadata(replacement))
            return replacement

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        with self._lock:
            return self._get(snapshot_id)

    def _get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        snapshot_path = self._find_snapshot_path(snapshot_id)
        if snapshot_path is None:
            return None
        return self._read_metadata(snapshot_path)

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        with self._lock:
            return self._list(model)

    def _list(self, model: str) -> list[ConfigSnapshotRecord]:
        model_root = self._model_root(model)
        if not model_root.exists():
            return []
        snapshots: list[ConfigSnapshotRecord] = []
        for snapshot_path in sorted(model_root.glob(f"*{SNAPSHOT_FILENAME_SUFFIX}")):
            resolved_path = self._resolve_existing_snapshot_path(snapshot_path)
            if resolved_path is None:
                continue
            snapshot = self._read_metadata(resolved_path)
            if snapshot is not None and snapshot.model == model:
                snapshots.append(snapshot)
        return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def list_all(self) -> list[ConfigSnapshotRecord]:
        with self._lock:
            return self._list_all()

    def _list_all(self) -> list[ConfigSnapshotRecord]:
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
        return sorted(snapshots, key=_snapshot_sort_key)

    def delete(self, snapshot_id: str) -> bool:
        with self._lock:
            snapshot_path = self._find_snapshot_path(snapshot_id)
            if snapshot_path is None:
                return False
            snapshot_path.unlink()
            return True

    def _snapshot_path(self, model: str, snapshot_id: str) -> Path:
        model_root = self._model_root(model)
        return resolve_under_root(
            self._root(),
            model_root / self._snapshot_filename(snapshot_id),
        )

    def _find_snapshot_path(self, snapshot_id: str) -> Path | None:
        filename = self._snapshot_filename(snapshot_id)
        if not self.root.exists():
            return None
        for candidate in sorted(self._root().rglob(f"*{SNAPSHOT_FILENAME_SUFFIX}")):
            if candidate.name != filename:
                continue
            snapshot_path = self._resolve_existing_snapshot_path(candidate)
            if snapshot_path is None:
                continue
            snapshot = self._read_metadata(snapshot_path)
            if snapshot is None or snapshot.id != snapshot_id:
                continue
            if not self._is_canonical_snapshot_path(snapshot_path, snapshot):
                continue
            return snapshot_path
        return None

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
        payload.update(model_identity_payload_from_id(snapshot.model))
    except ValueError:
        payload["model"] = snapshot.model
    return payload


def _record_from_metadata(payload: dict[str, object]) -> ConfigSnapshotRecord:
    overrides = payload["overrides"]
    if not isinstance(overrides, dict):
        raise TypeError("Config snapshot overrides must be a mapping.")
    model_id = model_id_from_payload(payload)
    if model_id is None:
        model_id = str(payload["model"])
    return ConfigSnapshotRecord(
        id=str(payload["id"]),
        model=model_id,
        preset=str(payload["preset"]),
        name=str(payload["name"]),
        overrides={str(key): str(value) for key, value in overrides.items()},
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
    )


def _snapshot_sort_key(snapshot: ConfigSnapshotRecord) -> tuple[str, str, str, str]:
    return (snapshot.model, snapshot.preset, snapshot.created_at, snapshot.id)


def _require_same_snapshot_identity(
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


def _require_no_snapshot_conflict(
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
