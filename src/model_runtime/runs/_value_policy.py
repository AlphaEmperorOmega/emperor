from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast


def deep_freeze(value: Any) -> Any:
    """Copy nested Runs values into recursively immutable containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {str(key): deep_freeze(item) for key, item in mapping.items()}
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(deep_freeze(item) for item in sequence)
    if isinstance(value, (set, frozenset)):
        values = cast(set[Any] | frozenset[Any], value)
        return frozenset(deep_freeze(item) for item in values)
    return value


def deep_thaw(value: Any) -> Any:
    """Project frozen Runs values into fresh ordinary wire containers."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, Any], value)
        return {key: deep_thaw(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        sequence = cast(tuple[Any, ...], value)
        return [deep_thaw(item) for item in sequence]
    if isinstance(value, frozenset):
        frozen_values = cast(frozenset[Any], value)
        return set(frozen_values)
    return value


__all__ = ["deep_freeze", "deep_thaw"]
