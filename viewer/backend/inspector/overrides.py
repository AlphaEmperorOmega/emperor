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
        if config_key is None:
            raise InspectorError(f"Unknown override '{raw_key}'.")
        try:
            parsed_value = parse_config_value(
                config_module,
                config_key,
                "" if raw_value is None else str(raw_value),
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
