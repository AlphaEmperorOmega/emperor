from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class NonFiniteJsonValue:
    path: str
    value: float


class NonFiniteJsonValueError(ValueError):
    def __init__(self, invalid: NonFiniteJsonValue) -> None:
        self.invalid = invalid
        super().__init__(f"JSON number at {invalid.path} must be finite.")


def _child_path(path: str, key: object) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    rendered = str(key)
    if rendered.isidentifier():
        return f"{path}.{rendered}"
    return f"{path}[{rendered!r}]"


def non_finite_json_values(
    value: Any,
    *,
    path: str = "$",
) -> tuple[NonFiniteJsonValue, ...]:
    """Return every non-finite float leaf with a stable JSON-style path."""

    invalid: list[NonFiniteJsonValue] = []

    def visit(current: Any, current_path: str) -> None:
        if isinstance(current, float):
            if not math.isfinite(current):
                invalid.append(NonFiniteJsonValue(current_path, current))
            return
        if isinstance(current, Mapping):
            mapping = cast(Mapping[object, Any], current)
            for key, child in mapping.items():
                visit(child, _child_path(current_path, key))
            return
        if isinstance(current, (list, tuple)):
            sequence = cast(list[Any] | tuple[Any, ...], current)
            for index, child in enumerate(sequence):
                visit(child, _child_path(current_path, index))

    visit(value, path)
    return tuple(invalid)


def require_finite_json(value: Any, *, path: str = "$") -> None:
    """Reject the first non-finite float in a JSON-shaped value."""

    invalid = non_finite_json_values(value, path=path)
    if invalid:
        raise NonFiniteJsonValueError(invalid[0])


def replace_non_finite_json(
    value: Any,
    *,
    path: str = "$",
    on_replace: Callable[[NonFiniteJsonValue], None] | None = None,
) -> Any:
    """Copy a JSON-shaped value while replacing non-finite floats with ``None``."""

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        invalid = NonFiniteJsonValue(path, value)
        if on_replace is not None:
            on_replace(invalid)
        return None
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {
            key: replace_non_finite_json(
                child,
                path=_child_path(path, key),
                on_replace=on_replace,
            )
            for key, child in mapping.items()
        }
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        return [
            replace_non_finite_json(
                child,
                path=_child_path(path, index),
                on_replace=on_replace,
            )
            for index, child in enumerate(sequence)
        ]
    if isinstance(value, tuple):
        sequence = cast(tuple[Any, ...], value)
        return tuple(
            replace_non_finite_json(
                child,
                path=_child_path(path, index),
                on_replace=on_replace,
            )
            for index, child in enumerate(sequence)
        )
    return value


__all__ = [
    "NonFiniteJsonValue",
    "NonFiniteJsonValueError",
    "non_finite_json_values",
    "replace_non_finite_json",
    "require_finite_json",
]
