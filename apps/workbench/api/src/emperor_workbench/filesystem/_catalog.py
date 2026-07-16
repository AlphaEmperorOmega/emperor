from __future__ import annotations

from pathlib import Path
from threading import Lock, RLock
from typing import Any

from emperor_workbench.filesystem._json import (
    read_json_object,
    write_json_atomic,
)
from emperor_workbench.filesystem._permissions import apply_owner_only_permissions

CATALOG_SCHEMA_VERSION = 1
CATALOG_DIRECTORY_MODE = 0o700
CATALOG_FILE_MODE = 0o600

_CATALOG_LOCKS_GUARD = Lock()
_CATALOG_LOCKS: dict[Path, RLock] = {}


def _catalog_lock(path: Path) -> RLock:
    with _CATALOG_LOCKS_GUARD:
        return _CATALOG_LOCKS.setdefault(path, RLock())


class PersistentJsonCatalog:
    """Atomic private catalog file scoped to one authoritative filesystem root."""

    def __init__(
        self,
        *,
        state_root: Path,
        name: str,
        authority_root: Path,
    ) -> None:
        private_state_root = self._private_directory(Path(state_root))
        catalog_root = self._private_directory(private_state_root / "catalogs")
        self.path = catalog_root / f"{name}.json"
        self.authority_root = str(Path(authority_root).resolve())
        self._lock = _catalog_lock(self.path)

    def load(self, *, kind: str) -> dict[str, Any] | None:
        with self._lock:
            if self.path.is_symlink():
                return None
            payload = read_json_object(self.path)
        if payload is None:
            return None
        if payload.get("schemaVersion") != CATALOG_SCHEMA_VERSION:
            return None
        if payload.get("kind") != kind:
            return None
        if payload.get("authorityRoot") != self.authority_root:
            return None
        if not isinstance(payload.get("generation"), int):
            return None
        return payload

    def publish(
        self,
        *,
        kind: str,
        generation: int,
        entries: list[dict[str, Any]],
    ) -> None:
        payload = {
            "schemaVersion": CATALOG_SCHEMA_VERSION,
            "kind": kind,
            "authorityRoot": self.authority_root,
            "generation": generation,
            "entries": entries,
        }
        with self._lock:
            write_json_atomic(self.path, payload)
            self.path.chmod(CATALOG_FILE_MODE)

    def invalidate(self) -> None:
        with self._lock:
            if self.path.is_symlink():
                return
            self.path.unlink(missing_ok=True)

    @staticmethod
    def _private_directory(path: Path) -> Path:
        if path.is_symlink():
            raise ValueError(f"Catalog directory must not be a symlink: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=CATALOG_DIRECTORY_MODE)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Catalog directory is not canonical: {path}")
        apply_owner_only_permissions(path)
        return path.resolve()
