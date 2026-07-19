import argparse
import inspect
from enum import Enum
from types import ModuleType, UnionType
from typing import Any, Union, get_args, get_origin


def _bool_value(raw_value: str) -> bool:
    value = raw_value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got '{raw_value}'")


def _none_value(raw_value: str) -> None:
    if raw_value.lower() in {"none", "null"}:
        return None
    raise argparse.ArgumentTypeError(f"expected none/null, got '{raw_value}'")


def _class_lookup(config_module: ModuleType, raw_value: str) -> type:
    value = raw_value.split(".")[-1]
    candidate = getattr(config_module, value, None)
    if inspect.isclass(candidate):
        return candidate
    raise argparse.ArgumentTypeError(f"unknown config class '{raw_value}'")


def _enum_lookup(enum_type: type[Enum], raw_value: str) -> Enum:
    value = raw_value.split(".")[-1]
    try:
        return enum_type[value]
    except KeyError as exc:
        choices = ", ".join(enum_type.__members__)
        raise argparse.ArgumentTypeError(
            f"unknown {enum_type.__name__} value '{raw_value}'. Choices: {choices}"
        ) from exc


def _annotation_classes(annotation: Any) -> list[type]:
    if annotation is None:
        return []
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {UnionType, Union}:
        return [item for arg in args for item in _annotation_classes(arg)]
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
    lowered = raw_value.lower()
    if lowered in {"none", "null"}:
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


def parse_config_value(
    config_module: ModuleType,
    key: str,
    raw_value: str,
) -> Any:
    current_value = getattr(config_module, key, None)
    if raw_value.lower() in {"none", "null"}:
        return None
    if isinstance(current_value, list):
        sample_value = next(
            (value for value in current_value if value is not None), None
        )
        if sample_value is None:
            if raw_value.lower() in {"none", "null"}:
                return None
            return raw_value
        temp_key = f"__{key}_ITEM"
        setattr(config_module, temp_key, sample_value)
        try:
            return parse_config_value(config_module, temp_key, raw_value)
        finally:
            delattr(config_module, temp_key)
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
        annotation = getattr(config_module, "__annotations__", {}).get(key)
        return _parse_from_annotation(config_module, annotation, raw_value)
    return raw_value
