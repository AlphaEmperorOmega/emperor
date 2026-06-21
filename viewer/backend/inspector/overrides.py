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

LEGACY_OVERRIDE_ALIASES = {
    "bias_flag": "stack_bias_flag",
    "hidden_dim": "stack_hidden_dim",
    "layer_norm_position": "stack_layer_norm_position",
    "gate_bias_flag": "gate_stack_bias_flag",
    "gate_hidden_dim": "gate_stack_hidden_dim",
    "gate_layer_norm_position": "gate_stack_layer_norm_position",
    "halting_bias_flag": "halting_stack_bias_flag",
    "halting_hidden_dim": "halting_stack_hidden_dim",
    "halting_layer_norm_position": "halting_stack_layer_norm_position",
    "memory_bias_flag": "memory_stack_bias_flag",
    "memory_hidden_dim": "memory_stack_hidden_dim",
    "memory_layer_norm_position": "memory_stack_layer_norm_position",
    "recurrent_gate_bias_flag": "recurrent_gate_stack_bias_flag",
    "recurrent_gate_hidden_dim": "recurrent_gate_stack_hidden_dim",
    "recurrent_gate_layer_norm_position": "recurrent_gate_stack_layer_norm_position",
    "recurrent_halting_bias_flag": "recurrent_halting_stack_bias_flag",
    "recurrent_halting_hidden_dim": "recurrent_halting_stack_hidden_dim",
    "recurrent_halting_layer_norm_position": (
        "recurrent_halting_stack_layer_norm_position"
    ),
}


def _legacy_override_alias(
    normalized_key: str,
    supported: Mapping[str, str],
) -> tuple[str | None, bool]:
    alias = LEGACY_OVERRIDE_ALIASES.get(normalized_key)
    if alias is not None:
        return supported.get(alias), False
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
