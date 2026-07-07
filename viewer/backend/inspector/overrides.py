from __future__ import annotations

from collections.abc import Mapping
from types import NoneType
from typing import Any, get_args

from models.config_overrides import (
    config_key_to_model_param,
    iter_supported_config_keys,
    normalize_key,
    parse_config_value,
)

from viewer.backend.inspector.config_classes import abstract_config_class_error
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.values import serialize_config_value


def supported_config_keys(config_module: Any) -> dict[str, str]:
    return {
        normalize_key(config_key): config_key
        for config_key in iter_supported_config_keys(config_module)
    }


def _supported_config_keys(config_module: Any) -> dict[str, str]:
    return supported_config_keys(config_module)


def resolve_override_key(
    normalized_key: str,
    supported: Mapping[str, str],
) -> tuple[str | None, bool]:
    config_key = supported.get(normalized_key)
    if config_key is not None:
        return config_key, False

    for config_key in supported.values():
        if config_key_to_model_param(config_key) == normalized_key:
            return config_key, False
    if normalized_key.endswith("_residual_flag"):
        residual_option_key = (
            normalized_key.removesuffix("_residual_flag")
            + "_residual_connection_option"
        )
        return supported.get(residual_option_key), True
    return None, False


def _legacy_residual_flag_value(raw_value: Any) -> str:
    normalized_value = "" if raw_value is None else str(raw_value).strip().lower()
    if normalized_value in {"true", "1", "yes", "y", "on"}:
        return "RESIDUAL"
    if normalized_value in {"false", "0", "no", "n", "off"}:
        return "DISABLED"
    return "" if raw_value is None else str(raw_value)


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


def _config_value_accepts_none(config_module: Any, config_key: str) -> bool:
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
    config_module: Any,
    config_key: str,
    raw_value: Any,
) -> str:
    if raw_value is None:
        if _config_value_accepts_none(config_module, config_key):
            return "None"
        return ""
    value = str(raw_value)
    if value == "" and _config_value_accepts_none(config_module, config_key):
        return "None"
    return value


def parse_override_mapping(
    config_module,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}

    supported = _supported_config_keys(config_module)
    parsed = {}
    for raw_key, raw_value in overrides.items():
        normalized_key = normalize_key(raw_key)
        config_key, legacy_residual_flag = resolve_override_key(
            normalized_key,
            supported,
        )
        if config_key is None:
            if ignore_unknown:
                continue
            raise InspectorError(f"Unknown override '{raw_key}'.")
        try:
            parse_value = (
                _legacy_residual_flag_value(raw_value)
                if legacy_residual_flag
                else _override_parse_value(config_module, config_key, raw_value)
            )
            parsed_value = parse_config_value(
                config_module,
                config_key,
                parse_value,
            )
            if isinstance(parsed_value, type):
                abstract_error = abstract_config_class_error(parsed_value)
                if abstract_error is not None:
                    raise ValueError(abstract_error)
            parsed[config_key_to_model_param(config_key)] = parsed_value
        except Exception as exc:
            raise InspectorError(
                f"Invalid value for override '{raw_key}': {raw_value!r}. {exc}"
            ) from exc
    return parsed


def canonicalize_override_keys(
    config_module,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}

    supported = _supported_config_keys(config_module)
    canonical: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        normalized_key = normalize_key(raw_key)
        config_key, legacy_residual_flag = resolve_override_key(
            normalized_key,
            supported,
        )
        if config_key is None:
            if ignore_unknown:
                continue
            raise InspectorError(f"Unknown override '{raw_key}'.")
        canonical[config_key] = (
            _legacy_residual_flag_value(raw_value)
            if legacy_residual_flag
            else raw_value
        )
    return canonical


def serialize_override_mapping(
    config_module,
    overrides: Mapping[str, Any] | None,
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    if not overrides:
        return {}

    supported = _supported_config_keys(config_module)
    serialized: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        normalized_key = normalize_key(raw_key)
        config_key, legacy_residual_flag = resolve_override_key(
            normalized_key,
            supported,
        )
        if config_key is None:
            if ignore_unknown:
                continue
            raise InspectorError(f"Unknown override '{raw_key}'.")
        parse_value = (
            _legacy_residual_flag_value(raw_value)
            if legacy_residual_flag
            else _override_parse_value(config_module, config_key, raw_value)
        )
        try:
            parsed_value = parse_config_value(config_module, config_key, parse_value)
        except Exception as exc:
            raise InspectorError(
                f"Invalid value for override '{raw_key}': {raw_value!r}. {exc}"
            ) from exc
        serialized[config_key] = serialize_config_value(parsed_value)
    return serialized
