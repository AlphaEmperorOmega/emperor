from __future__ import annotations

from collections.abc import Mapping
from types import NoneType
from typing import Any, get_args

from model_runtime.inspection.errors import InspectionError, _model_package_failure
from model_runtime.inspection.records import ParsedOverrides
from model_runtime.inspection.schema import preset_locks
from model_runtime.packages import (
    ModelPackage,
    abstract_config_class_error,
    config_key_to_model_param,
    iter_supported_config_keys,
    normalize_key,
    parse_config_value,
    serialize_config_value,
)


def _runtime_defaults(package: ModelPackage):
    try:
        return package.runtime_defaults
    except Exception as exc:
        raise _model_package_failure(package.catalog_key, exc) from exc


def supported_config_keys(package: ModelPackage) -> dict[str, str]:
    return {
        normalize_key(config_key): config_key
        for config_key in iter_supported_config_keys(_runtime_defaults(package))
    }


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


def _annotation_accepts_none(annotation: Any) -> bool:
    if annotation is None:
        return False
    if annotation is NoneType:
        return True
    if isinstance(annotation, str):
        return "None" in annotation or "Optional" in annotation
    if NoneType in get_args(annotation):
        return True
    return any(_annotation_accepts_none(arg) for arg in get_args(annotation))


def _config_value_accepts_none(package: ModelPackage, config_key: str) -> bool:
    config_module = _runtime_defaults(package)
    current_value = getattr(config_module, config_key, None)
    if current_value is None:
        return True
    if isinstance(current_value, list) and any(
        value is None for value in current_value
    ):
        return True
    annotation = getattr(config_module, "__annotations__", {}).get(config_key)
    return _annotation_accepts_none(annotation)


def _override_parse_value(
    package: ModelPackage,
    config_key: str,
    raw_value: Any,
) -> str:
    if raw_value is None:
        return "None" if _config_value_accepts_none(package, config_key) else ""
    value = str(raw_value)
    if value == "" and _config_value_accepts_none(package, config_key):
        return "None"
    return value


def reject_locked_overrides(
    package: ModelPackage,
    preset_name: str,
    parsed_overrides: Mapping[str, Any] | None,
) -> None:
    locks = preset_locks(package, preset_name)
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
    locks = preset_locks(package, preset_name)
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
    if not overrides:
        parsed: dict[str, Any] = {}
    else:
        supported = supported_config_keys(package)
        parsed = {}
        for raw_key, raw_value in overrides.items():
            normalized_key = normalize_key(raw_key)
            config_key = resolve_override_key(
                normalized_key,
                supported,
            )
            if config_key is None:
                if ignore_unknown:
                    continue
                raise InspectionError(f"Unknown override '{raw_key}'.")
            try:
                parsed_value = parse_config_value(
                    _runtime_defaults(package),
                    config_key,
                    _override_parse_value(package, config_key, raw_value),
                )
                if isinstance(parsed_value, type):
                    abstract_error = abstract_config_class_error(parsed_value)
                    if abstract_error is not None:
                        raise ValueError(abstract_error)
                parsed[config_key_to_model_param(config_key)] = parsed_value
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
    supported = supported_config_keys(package)
    canonical: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        config_key = resolve_override_key(
            normalize_key(raw_key),
            supported,
        )
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
    canonical = canonicalize_overrides(
        package,
        overrides,
        ignore_unknown=ignore_unknown,
    )
    serialized: dict[str, Any] = {}
    for config_key, raw_value in canonical.items():
        try:
            parsed = parse_config_value(
                _runtime_defaults(package),
                config_key,
                _override_parse_value(package, config_key, raw_value),
            )
        except InspectionError:
            raise
        except Exception as exc:
            raise InspectionError(
                f"Invalid value for override '{config_key}': {raw_value!r}. {exc}"
            ) from exc
        serialized[config_key] = serialize_config_value(parsed)
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
