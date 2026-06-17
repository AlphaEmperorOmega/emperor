"""Config snapshot library use cases.

A snapshot captures the non-default config overrides for a ``model + preset`` so
the variant can be reused and trained later. Validation mirrors the frontend
``validateConfigSnapshotCandidate`` in ``viewer/frontend/src/lib/config-snapshots.ts``
so the server (the source of truth) and the client agree on what is storable: at
least one non-default override, no preset-locked fields, and no duplicate of an
existing snapshot. Field defaults / locks come from the inspector rather than from
client input.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from models.catalog import model_identity_payload_from_id

from viewer.backend.config_snapshots import ConfigSnapshotRecord
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.repositories.config_snapshots import ConfigSnapshotRepository

IDENTITY_SEPARATOR = "\x00"


def config_schema(model: str, preset: str | None = None) -> dict[str, Any]:
    from viewer.backend.inspector.schema import config_schema as load_config_schema

    return load_config_schema(model, preset)


class ConfigSnapshotService:
    def __init__(self, repository: ConfigSnapshotRepository) -> None:
        self._repository = repository

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
        snapshot_name = self._validated_snapshot_name(
            model=model,
            preset=preset,
            name=name,
        )

        fields = config_schema(model, preset)["fields"]
        entries = self._validated_override_entries(
            model=model,
            preset=preset,
            fields=fields,
            overrides=overrides,
        )

        snapshot = ConfigSnapshotRecord(
            id=uuid.uuid4().hex,
            model=model,
            preset=preset,
            name=snapshot_name,
            overrides={entry["key"]: entry["value"] for entry in entries},
        )
        self._save_snapshot_record(snapshot)
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
        snapshot = self._require_snapshot(snapshot_id)
        if name is not None:
            snapshot.name = self._validated_snapshot_name(
                model=snapshot.model,
                preset=snapshot.preset,
                name=name,
                exclude_snapshot_id=snapshot.id,
            )
        if overrides is not None:
            fields = config_schema(snapshot.model, snapshot.preset)["fields"]
            entries = self._validated_override_entries(
                model=snapshot.model,
                preset=snapshot.preset,
                fields=fields,
                overrides=overrides,
                exclude_snapshot_id=snapshot.id,
            )
            snapshot.overrides = {entry["key"]: entry["value"] for entry in entries}
        snapshot.updated_at = _now()
        self._save_snapshot_record(snapshot)
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
            return self._repository.list_snapshots(model)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _list_all_snapshot_records(self) -> list[ConfigSnapshotRecord]:
        try:
            return self._repository.list_all_snapshots()
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _get_snapshot_record(
        self,
        snapshot_id: str,
    ) -> ConfigSnapshotRecord | None:
        try:
            return self._repository.get_snapshot(snapshot_id)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _save_snapshot_record(self, snapshot: ConfigSnapshotRecord) -> None:
        try:
            self._repository.save_snapshot(snapshot)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _delete_snapshot_record(self, snapshot_id: str) -> bool:
        try:
            return self._repository.delete_snapshot(snapshot_id)
        except ValueError as exc:
            raise InspectorError("Invalid config snapshot storage path.") from exc

    def _validated_snapshot_name(
        self,
        *,
        model: str,
        preset: str,
        name: str,
        exclude_snapshot_id: str | None = None,
    ) -> str:
        snapshot_name = name.strip()
        if not snapshot_name:
            raise InspectorError("Config snapshot name cannot be empty.")
        normalized_name = _normalize_snapshot_name(snapshot_name)
        for existing in self._list_snapshot_records(model):
            if existing.id == exclude_snapshot_id or existing.preset != preset:
                continue
            if _normalize_snapshot_name(existing.name) == normalized_name:
                raise InspectorError("A snapshot with this name already exists.")
        return snapshot_name

    def _validated_override_entries(
        self,
        *,
        model: str,
        preset: str,
        fields: list[dict[str, Any]],
        overrides: dict[str, str],
        exclude_snapshot_id: str | None = None,
    ) -> list[dict[str, str]]:
        entries, locked_fields = _override_entries(fields, overrides)
        if locked_fields:
            locked_names = ", ".join(
                str(field.get("label") or field["key"]) for field in locked_fields[:3]
            )
            raise InspectorError(
                f"Snapshots cannot include preset-locked fields: {locked_names}."
            )
        if not entries:
            raise InspectorError(
                "Change at least one non-default field before adding a snapshot."
            )

        identity = _identity(model, preset, entries)
        for existing in self._list_snapshot_records(model):
            if existing.id == exclude_snapshot_id or existing.preset != preset:
                continue
            existing_entries, _ = _override_entries(fields, existing.overrides)
            if _identity(model, preset, existing_entries) == identity:
                raise InspectorError(
                    "A snapshot with these config values already exists."
                )
        return entries


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _snapshot_to_api(snapshot: ConfigSnapshotRecord) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        **model_identity_payload_from_id(snapshot.model),
        "preset": snapshot.preset,
        "name": snapshot.name,
        "overrides": snapshot.overrides,
        "createdAt": snapshot.created_at,
        "updatedAt": snapshot.updated_at,
    }


def _override_entries(
    fields: list[dict[str, Any]],
    overrides: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    entries: list[dict[str, str]] = []
    locked_fields: list[dict[str, Any]] = []
    for field in fields:
        key = field["key"]
        if key not in overrides:
            continue
        if field.get("locked"):
            locked_fields.append(field)
        normalized = _normalize_value_for_field(field, overrides.get(key, ""))
        if normalized == _default_value_for_field(field):
            continue
        value = "" if field.get("nullable") and normalized == "null" else normalized
        entries.append({"key": key, "value": value})
    return entries, locked_fields


def _default_value_for_field(field: dict[str, Any]) -> str:
    return _normalize_value_for_field(field, field.get("default"))


def _normalize_value_for_field(field: dict[str, Any], value: Any) -> str:
    raw = "" if value is None else str(value).strip()
    if field.get("nullable") and raw == "":
        return "null"
    field_type = field.get("type")
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


def _to_number(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _normalize_snapshot_name(name: str) -> str:
    return name.strip().casefold()


def _identity(model: str, preset: str, entries: list[dict[str, str]]) -> str:
    parts = [model, preset, *(f"{entry['key']}={entry['value']}" for entry in entries)]
    return IDENTITY_SEPARATOR.join(parts)
