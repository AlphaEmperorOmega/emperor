import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Union, get_args, get_origin

from emperor.config import ConfigBase


@dataclass(frozen=True)
class _ConfigurationVisit:
    path: str
    config: ConfigBase
    expert_path: str | None
    row_scope: str
    restored_template: bool = False


def _config_validator(config: ConfigBase) -> type | None:
    try:
        return config.registry_owner().VALIDATOR
    except (AttributeError, NotImplementedError):
        return None


def _configuration_visits(
    value: object,
    path: str,
    ancestors: frozenset[int] | None = None,
    *,
    expert_path: str | None = None,
    row_scope: str = "",
    restored_template: bool = False,
) -> Iterator[_ConfigurationVisit]:
    """Visit occurrences, stopping cycles only along the current ancestry."""
    if not isinstance(value, (ConfigBase, dict, list, tuple)):
        return
    ancestors = frozenset() if ancestors is None else ancestors
    if id(value) in ancestors:
        return
    ancestors = ancestors | {id(value)}
    boundaries = {}
    if isinstance(value, ConfigBase):
        yield _ConfigurationVisit(
            path, value, expert_path, row_scope, restored_template
        )
        boundary_hook = getattr(
            _config_validator(value), "grouping_child_boundaries", None
        )
        if callable(boundary_hook):
            boundaries = boundary_hook(value)
        children = (
            (field.name, f"{path}.{field.name}", getattr(value, field.name))
            for field in fields(value)
        )
    elif isinstance(value, dict):
        children = ((key, f"{path}[{key!r}]", item) for key, item in value.items())
    else:
        children = (
            (index, f"{path}[{index}]", item) for index, item in enumerate(value)
        )
    for key, child_path, child in children:
        boundary = boundaries.get(key)
        yield from _configuration_visits(
            child,
            child_path,
            ancestors,
            expert_path=child_path if boundary == "expert" else expert_path,
            row_scope=child_path if boundary else row_scope,
            restored_template=boundary == "restored_template",
        )


def _adaptive_grouping_visits(
    config: object, *, root: str
) -> Iterator[_ConfigurationVisit]:
    for visit in _configuration_visits(config, root):
        enabled = getattr(_config_validator(visit.config), "grouping_is_enabled", None)
        if callable(enabled) and enabled(visit.config):
            yield visit


def _adaptive_grouping_configs(
    config: object,
    *,
    root: str,
    predicate: Callable[[ConfigBase], bool] | None = None,
    direct_only: bool = False,
) -> Iterator[tuple[str, ConfigBase]]:
    for visit in _adaptive_grouping_visits(config, root=root):
        if direct_only and visit.expert_path is not None:
            continue
        if predicate is None or predicate(visit.config):
            yield visit.path, visit.config


def _adaptive_grouping_paths(
    config: object,
    *,
    root: str,
    predicate: Callable[[ConfigBase], bool] | None = None,
    direct_only: bool = False,
) -> tuple[str, ...]:
    return tuple(
        path
        for path, _ in _adaptive_grouping_configs(
            config, root=root, predicate=predicate, direct_only=direct_only
        )
    )


def _first_adaptive_grouping_path(
    config: object,
    *,
    root: str,
    predicate: Callable[[ConfigBase], bool] | None = None,
    direct_only: bool = False,
) -> str | None:
    return next(
        (
            path
            for path, _ in _adaptive_grouping_configs(
                config, root=root, predicate=predicate, direct_only=direct_only
            )
        ),
        None,
    )


def _validate_adaptive_sequence_input(
    config: object,
    *,
    root: str,
    sequence_length: int,
    input_order: str,
) -> None:
    for path, augmentation in _adaptive_grouping_configs(
        config, root=root, direct_only=True
    ):
        augmentation.registry_owner().VALIDATOR.validate_grouping_sequence_input(
            augmentation,
            sequence_length=sequence_length,
            input_order=input_order,
            path=path,
        )


def _validate_grouped_row_preservation(config: object, *, root: str) -> None:
    scopes = {visit.row_scope for visit in _adaptive_grouping_visits(config, root=root)}
    if not scopes:
        return
    for visit in _configuration_visits(config, root):
        if visit.row_scope not in scopes or visit.restored_template:
            continue
        validate = getattr(
            _config_validator(visit.config), "validate_grouped_row_preservation", None
        )
        if callable(validate):
            validate(visit.config, path=visit.path)


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
