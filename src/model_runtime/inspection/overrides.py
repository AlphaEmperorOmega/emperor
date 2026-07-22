from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import ParsedOverrides
from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from model_runtime.packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_model_param,
    normalize_key,
)


def supported_config_keys(package: ModelPackage) -> dict[str, str]:
    spec = runtime_defaults_spec(package)
    return {normalize_key(config_key): config_key for config_key in spec.supported_keys}


def resolve_override_key(
    normalized_key: str,
    supported: Mapping[str, str],
) -> str | None:
    config_key = supported.get(normalized_key)
    if config_key is not None:
        return config_key
    for config_key in supported.values():
        if config_key_to_model_param(config_key) == normalized_key:
            return config_key
    return None


def reject_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any] | None,
) -> None:
    locks = runtime_defaults_spec(package).preset_locks(preset_name)
    locked_keys = sorted(set(parsed_overrides or {}) & set(locks))
    if not locked_keys:
        return
    details = ", ".join(
        f"{key} ({getattr(locks[key], 'reason', '')})" for key in locked_keys
    )
    raise InspectionError(
        f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
    )


def reject_conflicting_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any],
) -> None:
    locks = runtime_defaults_spec(package).preset_locks(preset_name)
    conflicts = sorted(
        key
        for key, value in parsed_overrides.items()
        if key in locks and value != getattr(locks[key], "value", None)
    )
    if not conflicts:
        return
    details = ", ".join(
        f"{key} ({getattr(locks[key], 'reason', '')})" for key in conflicts
    )
    raise InspectionError(
        f"Preset '{preset_name}' does not allow overriding locked fields: {details}"
    )


def parse_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    preset: str | None = None,
    ignore_unknown: bool = False,
) -> ParsedOverrides:
    spec = runtime_defaults_spec(package)
    if not overrides:
        parsed: dict[str, Any] = {}
    else:
        parsed = {}
        for raw_key, raw_value in overrides.items():
            config_key = spec.resolve_key(raw_key)
            if config_key is None:
                if ignore_unknown:
                    continue
                raise InspectionError(f"Unknown override '{raw_key}'.")
            try:
                parsed_value = spec.parse_value(config_key, raw_value)
                if isinstance(parsed_value, type):
                    abstract_error = abstract_config_class_error(parsed_value)
                    if abstract_error is not None:
                        raise ValueError(abstract_error)
                parsed[spec.model_parameter(config_key)] = parsed_value
            except InspectionError:
                raise
            except Exception as exc:
                raise InspectionError(
                    f"Invalid value for override '{raw_key}': {raw_value!r}. {exc}"
                ) from exc
    if preset is not None:
        reject_locked_overrides(package, preset, parsed)
    return ParsedOverrides(parsed)


def canonicalize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}
    spec = runtime_defaults_spec(package)
    canonical: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        config_key = spec.resolve_key(raw_key)
        if config_key is None:
            if ignore_unknown:
                continue
            raise InspectionError(f"Unknown override '{raw_key}'.")
        canonical[config_key] = raw_value
    return canonical


def serialize_overrides(
    package: ModelPackage,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    spec = runtime_defaults_spec(package)
    canonical = canonicalize_overrides(
        package,
        overrides,
        ignore_unknown=ignore_unknown,
    )
    serialized: dict[str, Any] = {}
    for config_key, raw_value in canonical.items():
        try:
            parsed = spec.parse_value(config_key, raw_value)
        except InspectionError:
            raise
        except Exception as exc:
            raise InspectionError(
                f"Invalid value for override '{config_key}': {raw_value!r}. {exc}"
            ) from exc
        serialized[config_key] = spec.serialize_value(parsed)
    return serialized


__all__ = [
    "canonicalize_overrides",
    "parse_overrides",
    "reject_conflicting_locked_overrides",
    "reject_locked_overrides",
    "resolve_override_key",
    "serialize_overrides",
    "supported_config_keys",
]
