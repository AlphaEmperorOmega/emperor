import types
from dataclasses import fields
from typing import Union, get_args, get_origin

from emperor.config import ConfigBase


def _adaptive_grouping_paths(
    config: ConfigBase,
    *,
    root: str,
) -> tuple[str, ...]:
    matches: list[str] = []
    _collect_adaptive_grouping_paths(config, root, set(), matches)
    return tuple(matches)


def _collect_adaptive_grouping_paths(
    value: object,
    path: str,
    visited: set[int],
    matches: list[str],
) -> None:
    if isinstance(value, ConfigBase):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        try:
            config_validator = value.registry_owner().VALIDATOR
        except (AttributeError, NotImplementedError):
            config_validator = None
        grouping_is_enabled = getattr(
            config_validator,
            "grouping_is_enabled",
            None,
        )
        if callable(grouping_is_enabled) and grouping_is_enabled(value):
            matches.append(path)
        for config_field in fields(value):
            field_value = getattr(value, config_field.name)
            field_path = f"{path}.{config_field.name}"
            _collect_adaptive_grouping_paths(
                field_value,
                field_path,
                visited,
                matches,
            )
        return

    if isinstance(value, dict):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        for key, item in value.items():
            _collect_adaptive_grouping_paths(
                item,
                f"{path}[{key!r}]",
                visited,
                matches,
            )
        return

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        for index, item in enumerate(value):
            _collect_adaptive_grouping_paths(
                item,
                f"{path}[{index}]",
                visited,
                matches,
            )


class ValidatorBase:
    OPTIONAL_FIELDS: set[str] = set()

    @classmethod
    def validate_required_fields(cls, cfg: ConfigBase) -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(
                    f"{field_name} is required for {cfg.__class__.__name__}, "
                    "received None"
                )

    @classmethod
    def validate_field_types(cls, cfg: ConfigBase) -> None:
        for field_name, field_info in cfg.__dataclass_fields__.items():
            if field_name in cls.OPTIONAL_FIELDS:
                continue
            value = getattr(cfg, field_name)
            if value is None:
                continue
            expected = cls._extract_type(field_info.type)
            if expected and (
                not isinstance(value, expected)
                or (expected is int and isinstance(value, bool))
            ):
                raise TypeError(
                    f"{field_name} must be {expected.__name__} for "
                    f"{cfg.__class__.__name__}, got {type(value).__name__}"
                )

    @classmethod
    def _extract_type(cls, annotation) -> type | None:
        if isinstance(annotation, type):
            return annotation
        origin = get_origin(annotation)
        if origin in (types.UnionType, Union):
            arguments = get_args(annotation)
            first_concrete = (
                arguments[1] if arguments[0] is type(None) else arguments[0]
            )
            return cls._extract_type(first_concrete)
        if isinstance(origin, type):
            return origin
        return None

    @staticmethod
    def validate_dimensions(**dims: int) -> None:
        for name, value in dims.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0, received {value}")
