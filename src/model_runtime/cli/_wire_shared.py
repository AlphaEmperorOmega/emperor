from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Sized
from typing import Any, cast

PROTOCOL_VERSION = 1
_MAX_JSON_NESTING_DEPTH = 64


class WireCodecError(ValueError):
    """A value does not conform to the version 1 CLI wire protocol."""


def require_sequence_limit(
    values: Sized,
    path: str,
    maximum_items: int,
) -> None:
    if len(values) > maximum_items:
        raise WireCodecError(f"{path} must contain at most {maximum_items} items.")


def wire_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WireCodecError(f"{path} must be an object.")
    mapping = cast(Mapping[object, Any], value)
    for key in mapping:
        if not isinstance(key, str):
            raise WireCodecError(f"{path} object keys must be strings.")
    return cast(Mapping[str, Any], mapping)


def wire_list(
    value: object,
    path: str,
    *,
    maximum_items: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise WireCodecError(f"{path} must be a list.")
    items = cast(list[Any], value)
    if maximum_items is not None:
        require_sequence_limit(items, path, maximum_items)
    return items


def wire_fields(
    value: object,
    *,
    path: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> Mapping[str, Any]:
    payload = wire_mapping(value, path)
    missing = [field for field in required if field not in payload]
    if missing:
        raise WireCodecError(f"{path} is missing required field {missing[0]!r}.")
    allowed = {*required, *optional}
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        raise WireCodecError(f"{path} contains unknown field {unknown[0]!r}.")
    return payload


def wire_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise WireCodecError(f"{path} must be a string.")
    return value


def wire_optional_string(value: object, path: str) -> str | None:
    return None if value is None else wire_string(value, path)


def wire_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise WireCodecError(f"{path} must be a boolean.")
    return value


def wire_int(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise WireCodecError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise WireCodecError(f"{path} must be at least {minimum}.")
    return value


def wire_optional_int(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
) -> int | None:
    return None if value is None else wire_int(value, path, minimum=minimum)


def wire_number(value: object, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireCodecError(f"{path} must be a number.")
    if isinstance(value, float) and not math.isfinite(value):
        raise WireCodecError(f"{path} must be finite.")
    return value


def wire_optional_number(value: object, path: str) -> int | float | None:
    return None if value is None else wire_number(value, path)


def wire_literal(value: object, path: str, allowed: set[str]) -> str:
    selected = wire_string(value, path)
    if selected not in allowed:
        expected = ", ".join(repr(item) for item in sorted(allowed))
        raise WireCodecError(f"{path} must be one of {expected}.")
    return selected


def wire_string_list(value: object, path: str) -> tuple[str, ...]:
    return tuple(
        wire_string(item, f"{path}[{index}]")
        for index, item in enumerate(wire_list(value, path))
    )


def wire_scalar(value: object, path: str) -> bool | int | float | str | None:
    if (
        value is None
        or type(value) is bool
        or type(value) is int
        or isinstance(value, str)
    ):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WireCodecError(f"{path} must be finite.")
        return value
    raise WireCodecError(f"{path} must be a JSON scalar.")


def wire_scalar_list(
    value: object,
    path: str,
) -> tuple[bool | int | float | str | None, ...]:
    return tuple(
        wire_scalar(item, f"{path}[{index}]")
        for index, item in enumerate(wire_list(value, path))
    )


def _require_json_container_depth(path: str, container_depth: int) -> None:
    if container_depth >= _MAX_JSON_NESTING_DEPTH:
        raise WireCodecError(
            f"{path} maximum JSON nesting depth of {_MAX_JSON_NESTING_DEPTH} exceeded."
        )


def json_value_to_wire(value: Any, *, path: str = "$") -> Any:
    """Project a JSON value with at most 64 nested container levels."""

    return _json_value_to_wire(
        value,
        path=path,
        container_depth=0,
        active_containers=set(),
    )


def _json_value_to_wire(
    value: Any,
    *,
    path: str,
    container_depth: int,
    active_containers: set[int],
) -> Any:
    if (
        value is None
        or type(value) is bool
        or type(value) is int
        or isinstance(value, str)
    ):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WireCodecError(f"{path} must be finite.")
        return value
    if isinstance(value, Mapping):
        payload = wire_mapping(cast(object, value), path)
        return _json_mapping_payload_to_wire(
            payload,
            path=path,
            container_depth=container_depth,
            active_containers=active_containers,
        )
    if isinstance(value, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        _require_json_container_depth(path, container_depth)
        container_id = id(cast(object, value))
        if container_id in active_containers:
            raise WireCodecError(f"{path} contains a cyclic JSON container reference.")
        active_containers.add(container_id)
        try:
            return [
                _json_value_to_wire(
                    item,
                    path=f"{path}[{index}]",
                    container_depth=container_depth + 1,
                    active_containers=active_containers,
                )
                for index, item in enumerate(sequence)
            ]
        finally:
            active_containers.remove(container_id)
    raise WireCodecError(f"{path} contains unsupported value {type(value).__name__}.")


def _json_mapping_payload_to_wire(
    payload: Mapping[str, Any],
    *,
    path: str,
    container_depth: int,
    active_containers: set[int],
) -> dict[str, Any]:
    _require_json_container_depth(path, container_depth)
    container_id = id(payload)
    if container_id in active_containers:
        raise WireCodecError(f"{path} contains a cyclic JSON container reference.")
    active_containers.add(container_id)
    try:
        return {
            key: _json_value_to_wire(
                item,
                path=f"{path}.{key}",
                container_depth=container_depth + 1,
                active_containers=active_containers,
            )
            for key, item in payload.items()
        }
    finally:
        active_containers.remove(container_id)


def json_value_from_wire(value: object, *, path: str = "$") -> Any:
    """Decode a JSON value with at most 64 nested container levels."""

    return _json_value_from_wire(
        value,
        path=path,
        container_depth=0,
        active_containers=set(),
    )


def _json_value_from_wire(
    value: object,
    *,
    path: str,
    container_depth: int,
    active_containers: set[int],
) -> Any:
    if (
        value is None
        or type(value) is bool
        or type(value) is int
        or isinstance(value, str)
    ):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WireCodecError(f"{path} must be finite.")
        return value
    if isinstance(value, Mapping):
        payload = wire_mapping(cast(object, value), path)
        return _json_mapping_payload_from_wire(
            payload,
            path=path,
            container_depth=container_depth,
            active_containers=active_containers,
        )
    if isinstance(value, list):
        sequence = cast(list[Any], value)
        _require_json_container_depth(path, container_depth)
        container_id = id(cast(object, value))
        if container_id in active_containers:
            raise WireCodecError(f"{path} contains a cyclic JSON container reference.")
        active_containers.add(container_id)
        try:
            return [
                _json_value_from_wire(
                    item,
                    path=f"{path}[{index}]",
                    container_depth=container_depth + 1,
                    active_containers=active_containers,
                )
                for index, item in enumerate(sequence)
            ]
        finally:
            active_containers.remove(container_id)
    raise WireCodecError(f"{path} contains unsupported value {type(value).__name__}.")


def _json_mapping_payload_from_wire(
    payload: Mapping[str, Any],
    *,
    path: str,
    container_depth: int,
    active_containers: set[int],
) -> dict[str, Any]:
    _require_json_container_depth(path, container_depth)
    container_id = id(payload)
    if container_id in active_containers:
        raise WireCodecError(f"{path} contains a cyclic JSON container reference.")
    active_containers.add(container_id)
    try:
        return {
            key: _json_value_from_wire(
                item,
                path=f"{path}.{key}",
                container_depth=container_depth + 1,
                active_containers=active_containers,
            )
            for key, item in payload.items()
        }
    finally:
        active_containers.remove(container_id)


def json_mapping_from_wire(value: object, *, path: str) -> dict[str, Any]:
    payload = wire_mapping(value, path)
    return _json_mapping_payload_from_wire(
        payload,
        path=path,
        container_depth=0,
        active_containers=set(),
    )


__all__ = [
    "PROTOCOL_VERSION",
    "WireCodecError",
    "json_mapping_from_wire",
    "json_value_from_wire",
    "json_value_to_wire",
    "wire_bool",
    "wire_fields",
    "wire_int",
    "wire_list",
    "wire_literal",
    "wire_mapping",
    "wire_number",
    "wire_optional_int",
    "wire_optional_number",
    "wire_optional_string",
    "wire_scalar",
    "wire_scalar_list",
    "wire_string",
    "wire_string_list",
]
