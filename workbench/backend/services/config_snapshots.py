"""Config snapshot library use cases.

A snapshot captures the non-default config overrides for a ``model + preset`` so
the variant can be reused and trained later. Validation mirrors the frontend
``validateConfigSnapshotCandidate`` in
``workbench/frontend/src/lib/config-snapshots.ts``
so the server (the source of truth) and the client agree on what is storable: at
least one non-default override, no preset-locked fields, and no duplicate of an
existing snapshot. Field defaults / locks come from the inspector rather than from
client input.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from emperor.inspection import (
    ConfigurationField,
    ConfigurationSchema,
    InspectionError,
    InspectionRequest,
)
from emperor.model_packages import (
    model_identity_payload_from_id,
    normalize_key,
)

from workbench.backend.config_snapshots import (
    ConfigSnapshotConflictError,
    ConfigSnapshotConflictReason,
    ConfigSnapshotRecord,
    ConfigSnapshotStore,
)
from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspector.errors import InspectorError


def config_schema(
    model: str,
    preset: str | None = None,
) -> ConfigurationSchema:
    return WorkbenchInspectionAdapter.select(model).configuration(preset)


class ConfigSnapshotService:
    def __init__(self, store: ConfigSnapshotStore) -> None:
        self._store = store

    def list_snapshots(self, model: str) -> list[dict[str, Any]]:
        return [
            _snapshot_to_api(snapshot)
            for snapshot in self._list_snapshot_records(model)
        ]

    def list_all_snapshots(self) -> list[dict[str, Any]]:
        return [
            _snapshot_to_api(snapshot)
            for snapshot in self._list_all_snapshot_records()
        ]

    def create_snapshot(
        self,
        *,
        model: str,
        preset: str,
        name: str,
        overrides: dict[str, str],
    ) -> dict[str, Any]:
        if not model or not preset:
            raise InspectorError("Select a model and preset first.")
        snapshot_name = self._validated_snapshot_name(name)
        fields = config_schema(model, preset).fields
        entries = self._validated_override_entries(
            model=model,
            preset=preset,
            fields=fields,
            overrides=overrides,
        )
        snapshot = self._create_snapshot_record(
            ConfigSnapshotRecord(
                id=uuid.uuid4().hex,
                model=model,
                preset=preset,
                name=snapshot_name,
                overrides={entry["key"]: entry["value"] for entry in entries},
            )
        )
        return _snapshot_to_api(snapshot)

    def rename_snapshot(self, snapshot_id: str, name: str) -> dict[str, Any]:
        return self.update_snapshot(snapshot_id, name=name)

    def update_snapshot(
        self,
        snapshot_id: str,
        *,
        name: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        current = self._require_snapshot(snapshot_id)
        snapshot_name = (
            self._validated_snapshot_name(name) if name is not None else current.name
        )
        snapshot_overrides = current.overrides
        if overrides is not None:
            fields = config_schema(current.model, current.preset).fields
            entries = self._validated_override_entries(
                model=current.model,
                preset=current.preset,
                fields=fields,
                overrides=overrides,
            )
            snapshot_overrides = {
                entry["key"]: entry["value"] for entry in entries
            }
        replacement = replace(
            current,
            name=snapshot_name,
            overrides=snapshot_overrides,
            updated_at=_now(),
        )
        snapshot = self._update_snapshot_record(current, replacement)
        if snapshot is None:
            raise InspectorError(f"Unknown config snapshot '{snapshot_id}'.")
        return _snapshot_to_api(snapshot)

    def delete_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        snapshot = self._require_snapshot(snapshot_id)
        self._delete_snapshot_record(snapshot_id)
        return {
            **model_identity_payload_from_id(snapshot.model),
            "snapshots": self.list_snapshots(snapshot.model),
        }

    def _require_snapshot(self, snapshot_id: str) -> ConfigSnapshotRecord:
        snapshot = self._get_snapshot_record(snapshot_id)
        if snapshot is None:
            raise InspectorError(f"Unknown config snapshot '{snapshot_id}'.")
        return snapshot

    def _list_snapshot_records(self, model: str) -> list[ConfigSnapshotRecord]:
        try:
            return self._store.list(model)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _list_all_snapshot_records(self) -> list[ConfigSnapshotRecord]:
        try:
            return self._store.list_all()
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _get_snapshot_record(
        self,
        snapshot_id: str,
    ) -> ConfigSnapshotRecord | None:
        try:
            return self._store.get(snapshot_id)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _create_snapshot_record(
        self,
        snapshot: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord:
        try:
            return self._store.create(snapshot)
        except ConfigSnapshotConflictError as exc:
            raise _snapshot_conflict_error(exc) from exc
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _update_snapshot_record(
        self,
        current: ConfigSnapshotRecord,
        replacement: ConfigSnapshotRecord,
    ) -> ConfigSnapshotRecord | None:
        try:
            return self._store.update(current, replacement)
        except ConfigSnapshotConflictError as exc:
            raise _snapshot_conflict_error(exc) from exc
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _delete_snapshot_record(self, snapshot_id: str) -> bool:
        try:
            return self._store.delete(snapshot_id)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _validated_snapshot_name(self, name: str) -> str:
        snapshot_name = name.strip()
        if not snapshot_name:
            raise InspectorError("Config snapshot name cannot be empty.")
        return snapshot_name

    def _validated_override_entries(
        self,
        *,
        model: str,
        preset: str,
        fields: tuple[ConfigurationField, ...],
        overrides: Mapping[str, str],
    ) -> list[dict[str, str]]:
        entries, locked_fields = _override_entries(fields, overrides)
        if locked_fields:
            locked_names = ", ".join(
                _field_label(field) for field in locked_fields[:3]
            )
            raise InspectorError(
                f"Snapshots cannot include preset-locked fields: {locked_names}."
            )
        if not entries:
            raise InspectorError(
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
    overrides = {entry["key"]: entry["value"] for entry in entries}
    adapter = WorkbenchInspectionAdapter.select(model)
    try:
        adapter.validate(
            InspectionRequest(preset=preset, overrides=overrides),
        )
    except InspectorError as exc:
        cause = exc.__cause__
        detail_source = cause if isinstance(cause, InspectionError) else exc
        raise InspectorError(
            "Invalid config snapshot overrides: "
            f"{_snapshot_config_error_detail(detail_source)}"
        ) from exc
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise InspectorError(f"Invalid config snapshot overrides: {detail}") from exc


def _snapshot_config_error_detail(exc: Exception) -> str:
    message = str(exc)
    cause = exc.__cause__
    if (
        message.startswith("Failed to build preset")
        and cause is not None
        and str(cause)
    ):
        return str(cause)
    return message or type(exc).__name__


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _snapshot_conflict_error(exc: ConfigSnapshotConflictError) -> InspectorError:
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
    return InspectorError(messages[exc.reason])


def _snapshot_to_api(snapshot: ConfigSnapshotRecord) -> dict[str, Any]:
    try:
        fields = config_schema(snapshot.model, snapshot.preset).fields
        overrides = _canonical_override_values(fields, snapshot.overrides)
    except Exception:
        overrides = dict(snapshot.overrides)
    return {
        "id": snapshot.id,
        **model_identity_payload_from_id(snapshot.model),
        "preset": snapshot.preset,
        "name": snapshot.name,
        "overrides": overrides,
        "createdAt": snapshot.created_at,
        "updatedAt": snapshot.updated_at,
    }


def _override_entries(
    fields: tuple[ConfigurationField, ...],
    overrides: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[ConfigurationField]]:
    entries: list[dict[str, str]] = []
    locked_fields: list[ConfigurationField] = []
    canonical_overrides = _canonical_override_values(fields, overrides)
    for field in fields:
        key = field.key
        if key not in canonical_overrides:
            continue
        if field.locked:
            locked_fields.append(field)
        normalized = _normalize_value_for_field(field, canonical_overrides.get(key, ""))
        if normalized == _default_value_for_field(field):
            continue
        value = "" if field.nullable and normalized == "null" else normalized
        entries.append({"key": key, "value": value})
    return entries, locked_fields


def _canonical_override_values(
    fields: tuple[ConfigurationField, ...],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    fields_by_key = _fields_by_override_key(fields)
    canonical: dict[str, str] = {}
    for raw_key, raw_value in overrides.items():
        field = fields_by_key.get(normalize_key(str(raw_key)))
        if field is None:
            continue
        key = field.key
        normalized = _normalize_value_for_field(field, raw_value)
        canonical[key] = (
            "" if field.nullable and normalized == "null" else normalized
        )
    return canonical


def _fields_by_override_key(
    fields: tuple[ConfigurationField, ...],
) -> dict[str, ConfigurationField]:
    fields_by_key: dict[str, ConfigurationField] = {}
    for field in fields:
        fields_by_key.setdefault(normalize_key(field.key), field)
    return fields_by_key


def _default_value_for_field(field: ConfigurationField) -> str:
    return _normalize_value_for_field(field, field.default)


def _normalize_value_for_field(field: ConfigurationField, value: Any) -> str:
    raw = "" if value is None else str(value).strip()
    if field.nullable and raw == "":
        return "null"
    field_type = field.value_type
    if field_type == "bool":
        lowered = raw.lower()
        if lowered in ("true", "false"):
            return lowered
    if field_type in ("int", "float"):
        number = _to_number(raw)
        if number is not None:
            if field_type == "int" and number.is_integer():
                return str(int(number))
            if field_type == "float":
                return str(int(number)) if number.is_integer() else str(number)
    return raw


def _field_label(field: ConfigurationField) -> str:
    return field.key.lower().replace("_", " ")


def _to_number(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
