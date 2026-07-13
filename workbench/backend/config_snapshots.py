from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock, RLock
from types import MappingProxyType
from typing import Any, Protocol

from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationSchema,
    InspectionRequest,
)
from model_runtime.packages import normalize_key
from workbench.backend.catalogs import PersistentJsonCatalog
from workbench.backend.failures import DomainFailure
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.model_identity import (
    model_id_from_payload,
    model_identity_payload_from_id,
)
from workbench.backend.mutation_context import deterministic_mutation_resource_id
from workbench.backend.storage.local_files import (
    read_json_object,
    require_safe_name,
    resolve_root,
    resolve_under_root,
    safe_child_path,
    write_json_atomic,
)

SNAPSHOT_FILENAME_SUFFIX = ".json"


class ConfigSnapshotFailure(DomainFailure):
    """A Config Snapshot request cannot be completed."""


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
                raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.STALE)
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
            _require_no_snapshot_conflict(snapshot, tuple(self._index.values()))
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
            _require_same_snapshot_identity(current, replacement)
            self._ensure_index(force=True)
            observed = self._get(current.id)
            if observed is None:
                return None
            if observed != current:
                raise ConfigSnapshotConflictError(ConfigSnapshotConflictReason.STALE)
            _require_no_snapshot_conflict(
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
            return self._list(model)

    def _list(self, model: str) -> list[ConfigSnapshotRecord]:
        self._model_root(model)
        self._ensure_index()
        assert self._index is not None
        snapshots = [
            snapshot for snapshot in self._index.values() if snapshot.model == model
        ]
        return sorted(snapshots, key=lambda snapshot: snapshot.created_at)

    def list_all(self) -> list[ConfigSnapshotRecord]:
        with self._lock:
            return self._list_all()

    def _list_all(self) -> list[ConfigSnapshotRecord]:
        self._ensure_index()
        assert self._index is not None
        return sorted(self._index.values(), key=_snapshot_sort_key)

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
        return sorted(snapshots, key=_snapshot_sort_key)

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

    def _snapshot_path(self, model: str, snapshot_id: str) -> Path:
        model_root = self._model_root(model)
        return resolve_under_root(
            self._root(),
            model_root / self._snapshot_filename(snapshot_id),
        )

    def _find_snapshot_path(self, snapshot_id: str) -> Path | None:
        self._snapshot_filename(snapshot_id)
        self._ensure_index()
        assert self._index is not None
        snapshot = self._index.get(snapshot_id)
        if snapshot is None:
            return None
        snapshot_path = self._snapshot_path(snapshot.model, snapshot.id)
        return self._resolve_existing_snapshot_path(snapshot_path)

    def _ensure_index(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self._index is None and not force:
            loaded = self._load_index()
            if loaded is not None:
                records, generation = loaded
                self._index = {record.id: record for record in records}
                self._index_generation = generation
                self._next_reconciliation = (
                    now + self._reconciliation_interval_seconds
                )
                return
        if (
            self._index is not None
            and not force
            and now < self._next_reconciliation
        ):
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
                for snapshot in sorted(self._index.values(), key=_snapshot_sort_key)
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


@dataclass(frozen=True, slots=True)
class ConfigSnapshotDeletion:
    model: str
    snapshots: tuple[ConfigSnapshotRecord, ...]


ConfigSnapshotSchemaSource = Callable[[str, str | None], ConfigurationSchema]


def config_snapshot_schema(
    model: str,
    preset: str | None = None,
) -> ConfigurationSchema:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    try:
        return WorkbenchInspectionAdapter.select(model).configuration(preset)
    except InspectionFailure as exc:
        raise ConfigSnapshotFailure(exc.detail) from exc


class ConfigSnapshotService:
    """Public semantic Interface for Config Snapshot use cases."""

    def __init__(
        self,
        store: ConfigSnapshotStore,
        *,
        schema_source: ConfigSnapshotSchemaSource | None = None,
    ) -> None:
        self._store = store
        self._schema_source = schema_source

    def _schema(self, model: str, preset: str | None) -> ConfigurationSchema:
        source = self._schema_source or config_snapshot_schema
        return source(model, preset)

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
    ) -> ConfigSnapshotRecord:
        if not model or not preset:
            raise ConfigSnapshotFailure("Select a model and preset first.")
        snapshot_id = deterministic_mutation_resource_id("config-snapshot")
        if snapshot_id is not None:
            existing = self.get_snapshot(snapshot_id)
            if existing is not None:
                return existing
        snapshot_name = _validated_snapshot_name(name)
        entries = self._validated_override_entries(
            model=model,
            preset=preset,
            overrides=overrides,
        )
        snapshot = ConfigSnapshotRecord(
            id=snapshot_id or uuid.uuid4().hex,
            model=model,
            preset=preset,
            name=snapshot_name,
            overrides={entry["key"]: entry["value"] for entry in entries},
        )
        try:
            return self._store.create(snapshot)
        except ConfigSnapshotConflictError as exc:
            raise _snapshot_conflict_error(exc) from exc
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
            entries = self._validated_override_entries(
                model=current.model,
                preset=current.preset,
                overrides=overrides,
            )
            snapshot_overrides = MappingProxyType(
                {entry["key"]: entry["value"] for entry in entries}
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
            raise _snapshot_conflict_error(exc) from exc
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
            fields = self._schema(snapshot.model, snapshot.preset).fields
            return MappingProxyType(
                _canonical_override_values(fields, snapshot.overrides)
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

    def _validated_override_entries(
        self,
        *,
        model: str,
        preset: str,
        overrides: Mapping[str, str],
    ) -> list[dict[str, str]]:
        fields = self._schema(model, preset).fields
        entries, locked_fields = _override_entries(fields, overrides)
        if locked_fields:
            locked_names = ", ".join(_field_label(field) for field in locked_fields[:3])
            raise ConfigSnapshotFailure(
                f"Snapshots cannot include preset-locked fields: {locked_names}."
            )
        if not entries:
            raise ConfigSnapshotFailure(
                "Change at least one non-default field before adding a snapshot."
            )
        _validate_snapshot_config(model=model, preset=preset, entries=entries)
        return entries


def _validate_snapshot_config(
    *,
    model: str,
    preset: str,
    entries: list[dict[str, str]],
) -> None:
    from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter

    overrides = {entry["key"]: entry["value"] for entry in entries}
    adapter = WorkbenchInspectionAdapter.select(model)
    try:
        adapter.validate(InspectionRequest(preset=preset, overrides=overrides))
    except InspectionFailure as exc:
        cause = exc.__cause__
        detail_source = cause if isinstance(cause, Exception) else exc
        raise ConfigSnapshotFailure(
            "Invalid config snapshot overrides: "
            f"{_snapshot_config_error_detail(detail_source)}"
        ) from exc
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise ConfigSnapshotFailure(
            f"Invalid config snapshot overrides: {detail}"
        ) from exc


def _snapshot_config_error_detail(exc: Exception) -> str:
    message = str(exc)
    remote_cause_detail = getattr(exc, "remote_cause_detail", None)
    if message.startswith("Failed to build preset") and isinstance(
        remote_cause_detail, str
    ):
        return remote_cause_detail
    cause = exc.__cause__
    if (
        message.startswith("Failed to build preset")
        and cause is not None
        and str(cause)
    ):
        return str(cause)
    return message or type(exc).__name__


def _snapshot_conflict_error(
    exc: ConfigSnapshotConflictError,
) -> ConfigSnapshotFailure:
    messages = {
        ConfigSnapshotConflictReason.ID: (
            "A config snapshot with this id already exists."
        ),
        ConfigSnapshotConflictReason.NAME: "A snapshot with this name already exists.",
        ConfigSnapshotConflictReason.RUNTIME_DEFAULTS: (
            "A snapshot with these config values already exists."
        ),
        ConfigSnapshotConflictReason.STALE: (
            "The config snapshot changed concurrently. Retry the update."
        ),
    }
    return ConfigSnapshotFailure(messages[exc.reason])


def _validated_snapshot_name(name: str) -> str:
    snapshot_name = name.strip()
    if not snapshot_name:
        raise ConfigSnapshotFailure("Config snapshot name cannot be empty.")
    return snapshot_name


def _override_entries(
    fields: tuple[ConfigurationField, ...],
    overrides: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[ConfigurationField]]:
    entries: list[dict[str, str]] = []
    locked_fields: list[ConfigurationField] = []
    canonical_overrides = _canonical_override_values(fields, overrides)
    for config_field in fields:
        if config_field.key not in canonical_overrides:
            continue
        if config_field.locked:
            locked_fields.append(config_field)
        normalized = _normalize_value_for_field(
            config_field,
            canonical_overrides.get(config_field.key, ""),
        )
        if normalized == _default_value_for_field(config_field):
            continue
        value = "" if config_field.nullable and normalized == "null" else normalized
        entries.append({"key": config_field.key, "value": value})
    return entries, locked_fields


def _canonical_override_values(
    fields: tuple[ConfigurationField, ...],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    fields_by_key = {normalize_key(field.key): field for field in fields}
    canonical: dict[str, str] = {}
    for raw_key, raw_value in overrides.items():
        field = fields_by_key.get(normalize_key(str(raw_key)))
        if field is None:
            continue
        normalized = _normalize_value_for_field(field, raw_value)
        canonical[field.key] = (
            "" if field.nullable and normalized == "null" else normalized
        )
    return canonical


def _default_value_for_field(field: ConfigurationField) -> str:
    return _normalize_value_for_field(field, field.default)


def _normalize_value_for_field(field: ConfigurationField, value: Any) -> str:
    raw = "" if value is None else str(value).strip()
    if field.nullable and raw == "":
        return "null"
    if field.value_type == "bool" and raw.lower() in ("true", "false"):
        return raw.lower()
    if field.value_type in ("int", "float"):
        try:
            number = float(raw)
        except (TypeError, ValueError):
            pass
        else:
            if field.value_type == "int" and number.is_integer():
                return str(int(number))
            if field.value_type == "float":
                return str(int(number)) if number.is_integer() else str(number)
    return raw


def _field_label(field: ConfigurationField) -> str:
    return field.key.lower().replace("_", " ")
