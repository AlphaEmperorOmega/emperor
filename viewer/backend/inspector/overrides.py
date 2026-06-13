from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from models.config_overrides import (
    config_key_to_model_param,
    iter_supported_config_keys,
    normalize_key,
    parse_config_value,
)

from viewer.backend.inspector.config_classes import abstract_config_class_error
from viewer.backend.inspector.errors import InspectorError


def _legacy_override_alias(
    normalized_key: str,
    supported: Mapping[str, str],
) -> tuple[str | None, bool]:
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


def parse_override_mapping(
    config_module, overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not overrides:
        return {}

    supported = {
        normalize_key(config_key): config_key
        for config_key in iter_supported_config_keys(config_module)
    }
    parsed = {}
    for raw_key, raw_value in overrides.items():
        normalized_key = normalize_key(raw_key)
        config_key = supported.get(normalized_key)
        legacy_residual_flag = False
        if config_key is None:
            config_key, legacy_residual_flag = _legacy_override_alias(
                normalized_key,
                supported,
            )
        if config_key is None:
            raise InspectorError(f"Unknown override '{raw_key}'.")
        try:
            parse_value = (
                _legacy_residual_flag_value(raw_value)
                if legacy_residual_flag
                else "" if raw_value is None else str(raw_value)
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
