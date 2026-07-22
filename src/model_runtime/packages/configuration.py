from __future__ import annotations

import inspect
from enum import Enum
from types import ModuleType, NoneType, UnionType
from typing import Any, Union, cast, get_args, get_origin

from emperor.config import ConfigBase

SKIP_CONFIG_KEYS = {
    "CONFIG_SCHEMA_SKIP_KEYS",
    "CONFIG_OVERRIDE_SKIP_KEYS",
    "DEFAULT_EXPERIMENT_TASK",
    "DATASET_OPTIONS_BY_TASK",
    "MONITOR_OPTIONS",
}


class ConfigValueError(ValueError):
    """A Runtime Defaults value cannot be interpreted through its public type."""


def normalize_key(key: str) -> str:
    return key.strip().replace("-", "_").lower()


def config_key_to_flag(key: str) -> str:
    return "--" + key.lower().replace("_", "-")


def config_key_to_param(key: str) -> str:
    return key.lower()


def config_key_to_model_param(key: str) -> str:
    return config_key_to_param(key)


def search_key_to_config_key(key: str) -> str:
    return "SEARCH_SPACE_" + normalize_key(key).upper()


def canonical_config_key(key: str) -> str:
    stripped_key = key.strip().replace("-", "_")
    if stripped_key.isupper():
        return stripped_key
    return normalize_key(key).upper()


def _is_supported_constant(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool, Enum, type))


def iter_supported_config_keys(config_module: ModuleType) -> list[str]:
    module_skip_keys = {
        key
        for key in getattr(config_module, "CONFIG_OVERRIDE_SKIP_KEYS", ())
        if isinstance(key, str)
    }
    skip_keys = SKIP_CONFIG_KEYS | module_skip_keys
    return sorted(
        key
        for key, value in vars(config_module).items()
        if not key.startswith("_")
        and key.isupper()
        and not key.startswith("SEARCH_SPACE_")
        and key not in skip_keys
        and _is_supported_constant(value)
    )


def _bool_value(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigValueError(f"expected a boolean value, got '{raw_value}'")


def _class_lookup(config_module: ModuleType, raw_value: str) -> type:
    value = raw_value.split(".")[-1]
    candidate = getattr(config_module, value, None)
    if inspect.isclass(candidate):
        return candidate
    raise ConfigValueError(f"unknown config class '{raw_value}'")


def _enum_lookup(enum_type: type[Enum], raw_value: str) -> Enum:
    value = raw_value.split(".")[-1]
    try:
        return enum_type[value]
    except KeyError as exc:
        choices = ", ".join(enum_type.__members__)
        raise ConfigValueError(
            f"unknown {enum_type.__name__} value '{raw_value}'. Choices: {choices}"
        ) from exc


def _annotation_classes(annotation: Any) -> list[type]:
    if annotation is None:
        return []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        return [item for arg in args for item in _annotation_classes(arg)]
    if annotation is NoneType:
        return []
    if origin is type and args and isinstance(args[0], type):
        return [args[0]]
    if inspect.isclass(annotation):
        return [annotation]
    return []


def _parse_from_annotation(
    config_module: ModuleType,
    annotation: Any,
    raw_value: str,
) -> Any:
    if raw_value.lower() in {"none", "null"}:
        return None
    classes = _annotation_classes(annotation)
    enum_classes = [cls for cls in classes if issubclass(cls, Enum)]
    if enum_classes:
        return _enum_lookup(enum_classes[0], raw_value)
    if bool in classes:
        return _bool_value(raw_value)
    if int in classes:
        return int(raw_value)
    if float in classes:
        return float(raw_value)
    if str in classes:
        return raw_value
    if classes:
        return _class_lookup(config_module, raw_value)
    return raw_value


def _parse_from_current_value(
    config_module: ModuleType,
    current_value: Any,
    annotation: Any,
    raw_value: str,
) -> Any:
    if raw_value.lower() in {"none", "null"}:
        return None
    if isinstance(current_value, list):
        values = cast(list[Any], current_value)
        sample = next((value for value in values if value is not None), None)
        if sample is None:
            return raw_value
        return _parse_from_current_value(config_module, sample, None, raw_value)
    if isinstance(current_value, bool):
        return _bool_value(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return raw_value
    if isinstance(current_value, Enum):
        return _enum_lookup(type(current_value), raw_value)
    if inspect.isclass(current_value):
        return _class_lookup(config_module, raw_value)
    if current_value is None:
        return _parse_from_annotation(config_module, annotation, raw_value)
    return raw_value


def parse_config_value(
    config_module: ModuleType,
    key: str,
    raw_value: str,
) -> Any:
    return _parse_from_current_value(
        config_module,
        getattr(config_module, key, None),
        getattr(config_module, "__annotations__", {}).get(key),
        raw_value,
    )


def abstract_config_class_error(candidate: type) -> str | None:
    try:
        if not issubclass(candidate, ConfigBase):
            return None
    except TypeError:
        return None
    try:
        instance = candidate()
    except TypeError:
        return None
    try:
        instance.registry_owner()
    except (NotImplementedError, ValueError) as exc:
        return str(exc)
    return None


def serialize_config_value(value: Any) -> bool | int | float | str | None:
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ConfigValueError",
    "SKIP_CONFIG_KEYS",
    "abstract_config_class_error",
    "canonical_config_key",
    "config_key_to_flag",
    "config_key_to_model_param",
    "config_key_to_param",
    "iter_supported_config_keys",
    "normalize_key",
    "parse_config_value",
    "search_key_to_config_key",
    "serialize_config_value",
]
