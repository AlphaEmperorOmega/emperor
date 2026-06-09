"""Config snapshot record storage interfaces and local adapters.

A config snapshot captures a named set of config overrides for a given
``model + preset`` so it can be reused and trained later. The Viewer backend has
no database, so persistence mirrors :mod:`viewer.backend.job_store`: records are
serialized to JSON on disk, one file per snapshot under ``<root>/<model>/<id>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from viewer.backend.storage.local_files import read_json_object, write_json_atomic

SNAPSHOT_FILENAME_SUFFIX = ".json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConfigSnapshotRecord:
    id: str
    model: str
    preset: str
    name: str
    overrides: dict[str, str]
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class ConfigSnapshotStore(Protocol):
    def save(self, snapshot: ConfigSnapshotRecord) -> None:
        ...

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        ...

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        ...

    def delete(self, snapshot_id: str) -> bool:
        ...


class InMemoryConfigSnapshotStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, ConfigSnapshotRecord] = {}

    @property
    def snapshots(self) -> dict[str, ConfigSnapshotRecord]:
        return self._snapshots

    def save(self, snapshot: ConfigSnapshotRecord) -> None:
        self._snapshots[snapshot.id] = snapshot

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        return self._snapshots.get(snapshot_id)

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        snapshots = [
            snapshot
            for snapshot in self._snapshots.values()
            if snapshot.model == model
        ]
        return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def delete(self, snapshot_id: str) -> bool:
        return self._snapshots.pop(snapshot_id, None) is not None


class FileSystemConfigSnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, snapshot: ConfigSnapshotRecord) -> None:
        snapshot_path = self._snapshot_path(snapshot.model, snapshot.id)
        write_json_atomic(snapshot_path, _record_to_metadata(snapshot))

    def get(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        snapshot_path = self._find_snapshot_path(snapshot_id)
        if snapshot_path is None:
            return None
        return self._read_metadata(snapshot_path)

    def list(self, model: str) -> list[ConfigSnapshotRecord]:
        model_root = self.root / model
        if not model_root.exists():
            return []
        snapshots: list[ConfigSnapshotRecord] = []
        for snapshot_path in sorted(model_root.glob(f"*{SNAPSHOT_FILENAME_SUFFIX}")):
            snapshot = self._read_metadata(snapshot_path)
            if snapshot is not None and snapshot.model == model:
                snapshots.append(snapshot)
        return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def delete(self, snapshot_id: str) -> bool:
        snapshot_path = self._find_snapshot_path(snapshot_id)
        if snapshot_path is None:
            return False
        snapshot_path.unlink()
        return True

    def _snapshot_path(self, model: str, snapshot_id: str) -> Path:
        return self.root / model / f"{snapshot_id}{SNAPSHOT_FILENAME_SUFFIX}"

    def _find_snapshot_path(self, snapshot_id: str) -> Path | None:
        if not self.root.exists():
            return None
        matches = sorted(self.root.rglob(f"{snapshot_id}{SNAPSHOT_FILENAME_SUFFIX}"))
        return matches[0] if matches else None

    def _read_metadata(self, snapshot_path: Path) -> ConfigSnapshotRecord | None:
        payload = read_json_object(snapshot_path)
        if payload is None:
            return None
        try:
            return _record_from_metadata(payload)
        except (KeyError, TypeError, ValueError):
            return None


def _record_to_metadata(snapshot: ConfigSnapshotRecord) -> dict[str, object]:
    return {
        "id": snapshot.id,
        "model": snapshot.model,
        "preset": snapshot.preset,
        "name": snapshot.name,
        "overrides": snapshot.overrides,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
    }


def _record_from_metadata(payload: dict[str, object]) -> ConfigSnapshotRecord:
    overrides = payload["overrides"]
    if not isinstance(overrides, dict):
        raise TypeError("Config snapshot overrides must be a mapping.")
    return ConfigSnapshotRecord(
        id=str(payload["id"]),
        model=str(payload["model"]),
        preset=str(payload["preset"]),
        name=str(payload["name"]),
        overrides={str(key): str(value) for key, value in overrides.items()},
        created_at=str(payload["created_at"]),
        updated_at=str(payload["updated_at"]),
    )
