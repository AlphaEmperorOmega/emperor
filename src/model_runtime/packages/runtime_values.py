from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType, ModuleType, NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from model_runtime.packages.configuration import (
    abstract_config_class_error,
    config_key_to_model_param,
    iter_supported_config_keys,
)


@dataclass(frozen=True, slots=True)
class RuntimeValueConstraints:
    """Package-declared semantic constraints for external Runtime Defaults."""

    positive: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        fields = cast(frozenset[object], self.positive)
        if any(not isinstance(field, str) or not field.isupper() for field in fields):
            raise ValueError(
                "Runtime value constraint fields must be uppercase config keys"
            )


def positive_runtime_fields(*config_keys: str) -> RuntimeValueConstraints:
    """Declare package-owned Runtime Defaults that must be strictly positive."""

    return RuntimeValueConstraints(positive=frozenset(config_keys))


_EMPTY_RUNTIME_VALUE_CONSTRAINTS = RuntimeValueConstraints()


def _runtime_default_mapping(value: object) -> dict[object, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Runtime Defaults values must be a mapping")
    return dict(cast(Mapping[object, object], value))


@cache
def _resolved_annotations(config_module: ModuleType) -> Mapping[str, Any]:
    return MappingProxyType(get_type_hints(config_module, include_extras=True))


def _runtime_annotation(config_module: ModuleType, config_key: str) -> Any | None:
    annotation = _resolved_annotations(config_module).get(config_key)
    if annotation is not None:
        return annotation
    current_value = getattr(config_module, config_key)
    if current_value is None:
        return None
    if inspect.isclass(current_value):
        return type
    return cast(Any, type(current_value))


@dataclass(frozen=True, slots=True)
class _RuntimeValueMismatch:
    path: str
    annotation: Any
    value: object
    tuple_length: int | None = None


@dataclass(frozen=True, slots=True)
class _InvalidRuntimeConfigClass:
    path: str
    candidate: type
    reason: str


def _is_empty_tuple_annotation(annotation: Any) -> bool:
    return annotation == tuple[()] or (
        get_origin(annotation) is tuple
        and not get_args(annotation)
        and hasattr(annotation, "__args__")
    )


def _transparent_annotation(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _first_unsupported_annotation(
    annotations: tuple[Any, ...],
    active: frozenset[int],
) -> Any | None:
    for annotation in annotations:
        unsupported = _unsupported_runtime_annotation(annotation, active)
        if unsupported is not None:
            return unsupported
    return None


def _is_nominal_runtime_class(annotation: Any) -> bool:
    if not inspect.isclass(annotation):
        return False
    annotation_metaclass = type(annotation)
    is_extensions_typed_dict = (
        annotation_metaclass.__module__ == "typing_extensions"
        and annotation_metaclass.__name__ == "_TypedDictMeta"
    )
    return (
        not is_typeddict(annotation)
        and not is_extensions_typed_dict
        and not bool(getattr(annotation, "_is_protocol", False))
    )


def _unsupported_runtime_annotation(
    annotation: Any,
    active: frozenset[int] = frozenset(),
) -> Any | None:
    if (
        annotation is Any
        or annotation is object
        or annotation is NoneType
        or annotation is type
        or _is_nominal_runtime_class(annotation)
    ):
        return None
    if id(annotation) in active:
        return annotation
    nested_active = active | {id(annotation)}
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {UnionType, Union} and arguments:
        unsupported = _first_unsupported_annotation(arguments, nested_active)
    elif origin is Annotated:
        unsupported = _unsupported_runtime_annotation(arguments[0], nested_active)
    elif origin is type:
        if len(arguments) != 1:
            unsupported = annotation
        else:
            expected_base = _transparent_annotation(arguments[0])
            unsupported = (
                None
                if expected_base is Any
                or expected_base is object
                or _is_nominal_runtime_class(expected_base)
                else annotation
            )
    elif origin is list:
        unsupported = (
            annotation
            if len(arguments) != 1
            else _unsupported_runtime_annotation(arguments[0], nested_active)
        )
    elif origin is tuple:
        unsupported = _unsupported_tuple_annotation(
            annotation,
            arguments,
            nested_active,
        )
    elif origin in {dict, Mapping}:
        unsupported = (
            annotation
            if len(arguments) != 2
            else _first_unsupported_annotation(arguments, nested_active)
        )
    else:
        unsupported = annotation
    return unsupported


def _unsupported_tuple_annotation(
    annotation: Any,
    arguments: tuple[Any, ...],
    active: frozenset[int],
) -> Any | None:
    if _is_empty_tuple_annotation(annotation):
        return None
    if not arguments:
        return annotation
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return _unsupported_runtime_annotation(arguments[0], active)
    if Ellipsis in arguments:
        return annotation
    return _first_unsupported_annotation(arguments, active)


def _list_value_mismatch(
    annotation: Any,
    value: object,
    path: str,
) -> _RuntimeValueMismatch | None:
    if not isinstance(value, list):
        return _RuntimeValueMismatch(path, annotation, value)
    item_annotation = get_args(annotation)[0]
    for index, item in enumerate(cast(list[object], value)):
        mismatch = _runtime_value_mismatch(
            item_annotation,
            item,
            f"{path}[{index}]",
        )
        if mismatch is not None:
            return mismatch
    return None


def _tuple_value_mismatch(
    annotation: Any,
    value: object,
    path: str,
) -> _RuntimeValueMismatch | None:
    if not isinstance(value, tuple):
        return _RuntimeValueMismatch(path, annotation, value)
    tuple_value = cast(tuple[object, ...], value)
    arguments = get_args(annotation)
    expected: tuple[Any, ...] = arguments
    if _is_empty_tuple_annotation(annotation):
        expected = ()
    elif len(arguments) == 2 and arguments[1] is Ellipsis:
        expected = (arguments[0],) * len(tuple_value)
    if len(tuple_value) != len(expected):
        return _RuntimeValueMismatch(
            path,
            annotation,
            tuple_value,
            len(tuple_value),
        )
    for index, (item, item_annotation) in enumerate(
        zip(tuple_value, expected, strict=True)
    ):
        mismatch = _runtime_value_mismatch(
            item_annotation,
            item,
            f"{path}[{index}]",
        )
        if mismatch is not None:
            return mismatch
    return None


def _mapping_value_mismatch(
    annotation: Any,
    value: object,
    path: str,
) -> _RuntimeValueMismatch | None:
    origin = get_origin(annotation)
    expected_origin = dict if origin is dict else Mapping
    if not isinstance(value, expected_origin):
        return _RuntimeValueMismatch(path, annotation, value)
    key_annotation, value_annotation = get_args(annotation)
    for index, (key, item) in enumerate(cast(Mapping[object, object], value).items()):
        key_mismatch = _runtime_value_mismatch(
            key_annotation,
            key,
            f"{path}.keys[{index}]",
        )
        if key_mismatch is not None:
            return key_mismatch
        value_mismatch = _runtime_value_mismatch(
            value_annotation,
            item,
            f"{path}.values[{index}]",
        )
        if value_mismatch is not None:
            return value_mismatch
    return None


def _direct_value_mismatch(
    annotation: Any,
    value: object,
    path: str,
) -> _RuntimeValueMismatch | None:
    mismatch: _RuntimeValueMismatch | None = _RuntimeValueMismatch(
        path,
        annotation,
        value,
    )
    if annotation is Any or annotation is object:
        mismatch = None
    elif annotation is NoneType and value is None:
        mismatch = None
    elif any(annotation is exact for exact in (bool, int, float, str)):
        mismatch = None if type(value) is annotation else mismatch
    elif annotation is type and inspect.isclass(value):
        mismatch = None
    elif _is_nominal_runtime_class(annotation) and isinstance(value, annotation):
        mismatch = None
    return mismatch


def _parameterized_value_mismatch(
    annotation: Any,
    value: object,
    path: str,
) -> _RuntimeValueMismatch | None:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {UnionType, Union}:
        matches = any(
            _runtime_value_mismatch(option, value, path) is None for option in arguments
        )
        mismatch = None if matches else _RuntimeValueMismatch(path, annotation, value)
    elif origin is Annotated:
        mismatch = _runtime_value_mismatch(arguments[0], value, path)
    elif origin is type:
        expected_base = _transparent_annotation(arguments[0])
        if inspect.isclass(value) and (
            expected_base is Any
            or expected_base is object
            or (
                _is_nominal_runtime_class(expected_base)
                and issubclass(value, expected_base)
            )
        ):
            mismatch = None
        else:
            mismatch = _RuntimeValueMismatch(path, annotation, value)
    elif origin is list:
        mismatch = _list_value_mismatch(annotation, value, path)
    elif origin is tuple:
        mismatch = _tuple_value_mismatch(annotation, value, path)
    elif origin in {dict, Mapping}:
        mismatch = _mapping_value_mismatch(annotation, value, path)
    else:
        mismatch = _RuntimeValueMismatch(path, annotation, value)
    return mismatch


def _runtime_value_mismatch(
    annotation: Any,
    value: object,
    path: str = "$",
) -> _RuntimeValueMismatch | None:
    if get_origin(annotation) is None:
        return _direct_value_mismatch(annotation, value, path)
    return _parameterized_value_mismatch(annotation, value, path)


def _container_annotation_pairs(
    annotation: Any,
    value: object,
) -> Iterable[tuple[Any, object]] | None:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is list and isinstance(value, list):
        return ((arguments[0], item) for item in cast(list[object], value))
    if origin is tuple and isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        item_annotations: tuple[Any, ...] = arguments
        if _is_empty_tuple_annotation(annotation):
            item_annotations = ()
        elif len(arguments) == 2 and arguments[1] is Ellipsis:
            item_annotations = (arguments[0],) * len(tuple_value)
        return zip(item_annotations, tuple_value, strict=True)
    return None


class _ConfigClassValidator:
    def __init__(self) -> None:
        self._checked: set[tuple[int, str]] = set()

    def invalid(
        self,
        annotation: Any,
        value: object,
        path: str = "$",
        *,
        root: bool = False,
    ) -> _InvalidRuntimeConfigClass | None:
        origin = get_origin(annotation)
        if root or origin is type or annotation is type:
            invalid = self._probe(value, path)
            if invalid is not None:
                return invalid
        if origin in {UnionType, Union}:
            return self._invalid_union(annotation, value, path)
        if origin is Annotated:
            return self.invalid(get_args(annotation)[0], value, path)
        if origin in {dict, Mapping} and isinstance(value, Mapping):
            return self._invalid_mapping(
                annotation,
                cast(Mapping[object, object], value),
                path,
            )
        return self._invalid_sequence(annotation, value, path)

    def _probe(
        self,
        value: object,
        path: str,
    ) -> _InvalidRuntimeConfigClass | None:
        if not inspect.isclass(value):
            return None
        identity = (id(value), path)
        if identity in self._checked:
            return None
        self._checked.add(identity)
        reason = abstract_config_class_error(value)
        if reason is None:
            return None
        return _InvalidRuntimeConfigClass(path, value, reason)

    def _invalid_union(
        self,
        annotation: Any,
        value: object,
        path: str,
    ) -> _InvalidRuntimeConfigClass | None:
        for option in get_args(annotation):
            if _runtime_value_mismatch(option, value, path) is not None:
                continue
            invalid = self.invalid(option, value, path)
            if invalid is not None:
                return invalid
        return None

    def _invalid_mapping(
        self,
        annotation: Any,
        value: Mapping[object, object],
        path: str,
    ) -> _InvalidRuntimeConfigClass | None:
        key_annotation, value_annotation = get_args(annotation)
        for index, (key, item) in enumerate(value.items()):
            invalid = self.invalid(
                key_annotation,
                key,
                f"{path}.keys[{index}]",
            ) or self.invalid(
                value_annotation,
                item,
                f"{path}.values[{index}]",
            )
            if invalid is not None:
                return invalid
        return None

    def _invalid_sequence(
        self,
        annotation: Any,
        value: object,
        path: str,
    ) -> _InvalidRuntimeConfigClass | None:
        pairs = _container_annotation_pairs(annotation, value)
        if pairs is None:
            return None
        for index, (item_annotation, item) in enumerate(pairs):
            invalid = self.invalid(item_annotation, item, f"{path}[{index}]")
            if invalid is not None:
                return invalid
        return None


def _parameterized_annotation_label(
    origin: Any,
    arguments: tuple[Any, ...],
) -> str | None:
    if origin in {UnionType, Union}:
        return " | ".join(_annotation_label(option) for option in arguments)
    if origin is type:
        if not arguments:
            return "type"
        return f"type[{_annotation_label(arguments[0])}]"
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        if not arguments:
            return name
        labels = ", ".join(_annotation_label(option) for option in arguments)
        return f"{name}[{labels}]"
    return None


def _annotation_label(annotation: Any) -> str:
    if annotation is NoneType:
        return "None"
    if annotation is Ellipsis:
        return "..."
    if _is_empty_tuple_annotation(annotation):
        return "tuple[()]"
    parameterized = _parameterized_annotation_label(
        get_origin(annotation),
        get_args(annotation),
    )
    if parameterized is not None:
        return parameterized
    return getattr(annotation, "__name__", str(annotation))


def _unsupported_annotation_error(
    *,
    package: str,
    key: str,
    annotation: Any,
) -> TypeError:
    return TypeError(
        f"{package}: runtime key {key!r} declares unsupported annotation "
        f"{_annotation_label(annotation)}"
    )


def _runtime_value_type_error(
    *,
    package: str,
    key: str,
    mismatch: _RuntimeValueMismatch,
) -> TypeError:
    location = "" if mismatch.path == "$" else f" at {mismatch.path}"
    expected = _annotation_label(mismatch.annotation)
    if mismatch.tuple_length is not None:
        return TypeError(
            f"{package}: runtime key {key!r}{location} has tuple length "
            f"{mismatch.tuple_length}; expected {expected}"
        )
    return TypeError(
        f"{package}: runtime key {key!r}{location} has type "
        f"{type(mismatch.value).__name__}; expected {expected}"
    )


def _invalid_config_class_error(
    *,
    package: str,
    key: str,
    invalid: _InvalidRuntimeConfigClass,
) -> ValueError:
    location = "" if invalid.path == "$" else f" at {invalid.path}"
    return ValueError(
        f"{package}: runtime key {key!r}{location} selects invalid config class "
        f"{invalid.candidate.__name__}: {invalid.reason}"
    )


def _runtime_keys_by_alias(config_module: ModuleType) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for config_key in iter_supported_config_keys(config_module):
        aliases[config_key.lower()] = config_key
        aliases[config_key_to_model_param(config_key)] = config_key
    return aliases


def _declared_runtime_value_constraints(
    config_module: ModuleType,
) -> RuntimeValueConstraints:
    constraints = getattr(
        config_module,
        "RUNTIME_VALUE_CONSTRAINTS",
        _EMPTY_RUNTIME_VALUE_CONSTRAINTS,
    )
    if not isinstance(constraints, RuntimeValueConstraints):
        raise TypeError(
            f"{config_module.__name__}.RUNTIME_VALUE_CONSTRAINTS must be a "
            "RuntimeValueConstraints value"
        )
    supported = set(iter_supported_config_keys(config_module))
    declared = constraints.positive
    unknown = sorted(declared - supported)
    if unknown:
        raise ValueError(
            f"{config_module.__name__}.RUNTIME_VALUE_CONSTRAINTS references "
            f"unknown config key(s): {', '.join(unknown)}"
        )
    return constraints


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _validate_runtime_value_constraints(
    values: Mapping[str, object],
    *,
    package: str,
    config_module: ModuleType,
    accepted: Mapping[str, str],
) -> None:
    constraints = _declared_runtime_value_constraints(config_module)
    for runtime_key, value in values.items():
        config_key = accepted[runtime_key]
        number = _finite_number(value)
        if config_key in constraints.positive and (number is None or number <= 0):
            raise ValueError(
                f"{package}: runtime key {runtime_key!r} must be positive; "
                f"got {value!r}"
            )


def validate_runtime_default_value_types(
    values: Mapping[str, object],
    *,
    package: str,
    config_module: ModuleType,
) -> None:
    """Validate known overrides against their package-owned declared types.

    Unknown keys remain the responsibility of the package adapter so this shared
    boundary does not change package-local alias or error-order semantics.
    """

    accepted = _runtime_keys_by_alias(config_module)
    if any(key not in accepted for key in values):
        return
    for key, value in values.items():
        config_key = accepted[key]
        annotation = _runtime_annotation(config_module, config_key)
        if annotation is None:
            continue
        unsupported = _unsupported_runtime_annotation(annotation)
        if unsupported is not None:
            raise _unsupported_annotation_error(
                package=package,
                key=key,
                annotation=unsupported,
            )
        mismatch = _runtime_value_mismatch(annotation, value)
        if mismatch is not None:
            raise _runtime_value_type_error(
                package=package,
                key=key,
                mismatch=mismatch,
            )
        invalid = _ConfigClassValidator().invalid(
            annotation,
            value,
            root=True,
        )
        if invalid is not None:
            raise _invalid_config_class_error(
                package=package,
                key=key,
                invalid=invalid,
            )
    _validate_runtime_value_constraints(
        values,
        package=package,
        config_module=config_module,
        accepted=accepted,
    )


def validate_runtime_default_values(
    values: Mapping[str, object] | None,
    *,
    package: str,
    config_module: ModuleType,
) -> dict[str, object]:
    try:
        resolved = _runtime_default_mapping(values)
    except TypeError as exc:
        raise TypeError(f"{package}: {exc}") from exc
    if any(not isinstance(key, str) for key in resolved):
        raise TypeError(f"{package}: Runtime Defaults keys must be strings")
    normalized = {cast(str, key): value for key, value in resolved.items()}
    accepted = set(_runtime_keys_by_alias(config_module))
    unknown = sorted(set(normalized) - accepted)
    if unknown:
        fields = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{package}: unknown Runtime Defaults field(s): {fields}")
    validate_runtime_default_value_types(
        normalized,
        package=package,
        config_module=config_module,
    )
    return normalized


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeOptions:
    """Immutable package-local construction values produced from flat defaults."""

    _values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def _as_construction_kwargs(self) -> dict[str, object]:
        return dict(self._values)


__all__ = [
    "ResolvedRuntimeOptions",
    "RuntimeValueConstraints",
    "positive_runtime_fields",
    "validate_runtime_default_value_types",
    "validate_runtime_default_values",
]
