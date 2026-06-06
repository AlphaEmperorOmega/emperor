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
from datetime import datetime, timezone
from typing import Any

from viewer.backend.config_snapshots import ConfigSnapshotRecord
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.schema import config_schema
from viewer.backend.repositories.config_snapshots import ConfigSnapshotRepository

MAX_DEFAULT_NAME_FIELDS = 3
IDENTITY_SEPARATOR = "\x00"


class ConfigSnapshotService:
    def __init__(self, repository: ConfigSnapshotRepository) -> None:
        self._repository = repository

    def list_snapshots(self, model: str) -> list[dict[str, Any]]:
        return [
            _snapshot_to_api(snapshot)
            for snapshot in self._repository.list_snapshots(model)
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

        fields = config_schema(model, preset)["fields"]
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
        for existing in self._repository.list_snapshots(model):
            if existing.preset != preset:
                continue
            existing_entries, _ = _override_entries(fields, existing.overrides)
            if _identity(model, preset, existing_entries) == identity:
                raise InspectorError(
                    "A snapshot with these config values already exists."
                )

        snapshot = ConfigSnapshotRecord(
            id=uuid.uuid4().hex,
            model=model,
            preset=preset,
            name=name.strip() or _default_name(preset, entries),
            overrides={entry["key"]: entry["value"] for entry in entries},
        )
        self._repository.save_snapshot(snapshot)
        return _snapshot_to_api(snapshot)

    def rename_snapshot(self, snapshot_id: str, name: str) -> dict[str, Any]:
        snapshot = self._require_snapshot(snapshot_id)
        new_name = name.strip()
        if not new_name:
            raise InspectorError("Config snapshot name cannot be empty.")
        snapshot.name = new_name
        snapshot.updated_at = _now()
        self._repository.save_snapshot(snapshot)
        return _snapshot_to_api(snapshot)

    def delete_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        snapshot = self._require_snapshot(snapshot_id)
        self._repository.delete_snapshot(snapshot_id)
        return {
            "model": snapshot.model,
            "snapshots": self.list_snapshots(snapshot.model),
        }

    def _require_snapshot(self, snapshot_id: str) -> ConfigSnapshotRecord:
        snapshot = self._repository.get_snapshot(snapshot_id)
        if snapshot is None:
            raise InspectorError(f"Unknown config snapshot '{snapshot_id}'.")
        return snapshot


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_to_api(snapshot: ConfigSnapshotRecord) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "model": snapshot.model,
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


def _identity(model: str, preset: str, entries: list[dict[str, str]]) -> str:
    parts = [model, preset, *(f"{entry['key']}={entry['value']}" for entry in entries)]
    return IDENTITY_SEPARATOR.join(parts)


def _default_name(preset: str, entries: list[dict[str, str]]) -> str:
    if not entries:
        return f"{preset or 'config'} snapshot"
    visible = entries[:MAX_DEFAULT_NAME_FIELDS]
    suffix = (
        f" +{len(entries) - MAX_DEFAULT_NAME_FIELDS}"
        if len(entries) > MAX_DEFAULT_NAME_FIELDS
        else ""
    )
    body = " ".join(
        f"{entry['key']}={entry['value'] or 'None'}" for entry in visible
    )
    return f"{preset} {body}{suffix}"
