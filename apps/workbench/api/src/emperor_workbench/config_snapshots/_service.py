from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Self

from emperor_workbench.config_snapshots import _validation
from emperor_workbench.config_snapshots._errors import (
    ConfigSnapshotConflictError,
    ConfigSnapshotFailure,
    config_snapshot_conflict_failure,
)
from emperor_workbench.config_snapshots._filesystem_store import (
    FileSystemConfigSnapshotStore,
)
from emperor_workbench.config_snapshots._memory_store import (
    InMemoryConfigSnapshotStore,
)
from emperor_workbench.config_snapshots._records import (
    ConfigSnapshotDeletion,
    ConfigSnapshotRecord,
)
from emperor_workbench.config_snapshots._store import ConfigSnapshotStore
from emperor_workbench.model_packages import ModelPackageCatalog


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConfigSnapshotService:
    """Semantic operations for Workbench-owned Config Snapshots."""

    def __init__(
        self,
        store: ConfigSnapshotStore,
        *,
        model_packages: ModelPackageCatalog,
    ) -> None:
        self._store = store
        self._model_packages = model_packages

    @classmethod
    def in_memory(
        cls,
        *,
        model_packages: ModelPackageCatalog,
    ) -> Self:
        return cls(
            InMemoryConfigSnapshotStore(),
            model_packages=model_packages,
        )

    @classmethod
    def from_filesystem(
        cls,
        root: Path,
        *,
        model_packages: ModelPackageCatalog,
        state_root: Path | None = None,
        reconciliation_interval_seconds: float = 30.0,
    ) -> Self:
        return cls(
            FileSystemConfigSnapshotStore(
                root,
                state_root=state_root,
                reconciliation_interval_seconds=reconciliation_interval_seconds,
            ),
            model_packages=model_packages,
        )

    def list_snapshots(self, model: str) -> tuple[ConfigSnapshotRecord, ...]:
        try:
            return tuple(self._store.list(model))
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc

    def list_all_snapshots(self) -> tuple[ConfigSnapshotRecord, ...]:
        try:
            return tuple(self._store.list_all())
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc

    def get_snapshot(self, snapshot_id: str) -> ConfigSnapshotRecord | None:
        try:
            return self._store.get(snapshot_id)
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc

    def require_snapshot(self, snapshot_id: str) -> ConfigSnapshotRecord:
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ConfigSnapshotFailure(f"Unknown config snapshot '{snapshot_id}'.")
        return snapshot

    def create_snapshot(
        self,
        *,
        model: str,
        preset: str,
        name: str,
        overrides: Mapping[str, str],
        snapshot_id: str | None = None,
    ) -> ConfigSnapshotRecord:
        if not model or not preset:
            raise ConfigSnapshotFailure("Select a model and preset first.")
        if snapshot_id is not None:
            existing = self.get_snapshot(snapshot_id)
            if existing is not None:
                return existing
        snapshot_name = _validated_snapshot_name(name)
        canonical_overrides = _validation.validated_overrides(
            self._model_packages,
            model=model,
            preset=preset,
            overrides=overrides,
        )
        snapshot = ConfigSnapshotRecord(
            id=snapshot_id or uuid.uuid4().hex,
            model=model,
            preset=preset,
            name=snapshot_name,
            overrides=canonical_overrides,
        )
        try:
            return self._store.create(snapshot)
        except ConfigSnapshotConflictError as exc:
            raise config_snapshot_conflict_failure(exc) from exc
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc

    def rename_snapshot(self, snapshot_id: str, name: str) -> ConfigSnapshotRecord:
        return self.update_snapshot(snapshot_id, name=name)

    def update_snapshot(
        self,
        snapshot_id: str,
        *,
        name: str | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> ConfigSnapshotRecord:
        current = self.require_snapshot(snapshot_id)
        snapshot_name = (
            _validated_snapshot_name(name) if name is not None else current.name
        )
        snapshot_overrides = current.overrides
        if overrides is not None:
            snapshot_overrides = MappingProxyType(
                _validation.validated_overrides(
                    self._model_packages,
                    model=current.model,
                    preset=current.preset,
                    overrides=overrides,
                )
            )
        replacement = replace(
            current,
            name=snapshot_name,
            overrides=snapshot_overrides,
            updated_at=_now(),
        )
        try:
            snapshot = self._store.update(current, replacement)
        except ConfigSnapshotConflictError as exc:
            raise config_snapshot_conflict_failure(exc) from exc
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc
        if snapshot is None:
            raise ConfigSnapshotFailure(f"Unknown config snapshot '{snapshot_id}'.")
        return snapshot

    def delete_snapshot(self, snapshot_id: str) -> ConfigSnapshotDeletion:
        snapshot = self.require_snapshot(snapshot_id)
        try:
            self._store.delete(snapshot_id)
        except ValueError as exc:
            raise ConfigSnapshotFailure(
                "Invalid config snapshot storage path."
            ) from exc
        return ConfigSnapshotDeletion(
            model=snapshot.model,
            snapshots=self.list_snapshots(snapshot.model),
        )

    def canonical_overrides(
        self,
        snapshot: ConfigSnapshotRecord,
    ) -> Mapping[str, str]:
        try:
            return MappingProxyType(
                _validation.canonical_overrides(
                    self._model_packages,
                    model=snapshot.model,
                    preset=snapshot.preset,
                    overrides=snapshot.overrides,
                )
            )
        except Exception:
            return snapshot.overrides

    def semantic_revision(self, snapshot: ConfigSnapshotRecord) -> str:
        semantic = {
            "model": snapshot.model,
            "preset": snapshot.preset,
            "overrides": dict(self.canonical_overrides(snapshot)),
        }
        encoded = json.dumps(
            semantic,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _validated_snapshot_name(name: str) -> str:
    snapshot_name = name.strip()
    if not snapshot_name:
        raise ConfigSnapshotFailure("Config snapshot name cannot be empty.")
    return snapshot_name


__all__ = ["ConfigSnapshotService"]
