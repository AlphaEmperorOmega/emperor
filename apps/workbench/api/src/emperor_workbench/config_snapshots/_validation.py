from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationSchema,
    InspectionRequest,
)
from model_runtime.packages import normalize_key

from emperor_workbench.config_snapshots._errors import ConfigSnapshotFailure
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    ModelPackageFailure,
)


def configuration_schema(
    model_packages: ModelPackageCatalog,
    model: str,
    preset: str | None = None,
) -> ConfigurationSchema:
    try:
        return model_packages.select(model).configuration(preset)
    except ModelPackageFailure as exc:
        raise ConfigSnapshotFailure(exc.detail) from exc


def validate_config_snapshot(
    model_packages: ModelPackageCatalog,
    model: str,
    preset: str,
    entries: list[dict[str, str]],
) -> None:
    overrides = {entry["key"]: entry["value"] for entry in entries}
    try:
        model_packages.select(model).validate(
            InspectionRequest(preset=preset, overrides=overrides)
        )
    except ModelPackageFailure as exc:
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


def validated_overrides(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    preset: str,
    overrides: Mapping[str, str],
) -> dict[str, str]:
    fields = configuration_schema(model_packages, model, preset).fields
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
    validate_config_snapshot(model_packages, model, preset, entries)
    return {entry["key"]: entry["value"] for entry in entries}


def canonical_overrides(
    model_packages: ModelPackageCatalog,
    *,
    model: str,
    preset: str,
    overrides: Mapping[str, str],
) -> dict[str, str]:
    fields = configuration_schema(model_packages, model, preset).fields
    return _canonical_override_values(fields, overrides)


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


def _override_entries(
    fields: tuple[ConfigurationField, ...],
    overrides: Mapping[str, str],
) -> tuple[list[dict[str, str]], list[ConfigurationField]]:
    entries: list[dict[str, str]] = []
    locked_fields: list[ConfigurationField] = []
    canonical = _canonical_override_values(fields, overrides)
    for config_field in fields:
        if config_field.key not in canonical:
            continue
        if config_field.locked:
            locked_fields.append(config_field)
        normalized = _normalize_value_for_field(
            config_field,
            canonical.get(config_field.key, ""),
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
    raw_value = "" if value is None else str(value).strip()
    if field.nullable and raw_value == "":
        return "null"
    if field.value_type == "bool" and raw_value.lower() in ("true", "false"):
        return raw_value.lower()
    if field.value_type in ("int", "float"):
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            pass
        else:
            if field.value_type == "int" and number.is_integer():
                return str(int(number))
            if field.value_type == "float":
                return str(int(number)) if number.is_integer() else str(number)
    return raw_value


def _field_label(field: ConfigurationField) -> str:
    return field.key.lower().replace("_", " ")


__all__ = [
    "canonical_overrides",
    "configuration_schema",
    "validate_config_snapshot",
    "validated_overrides",
]
