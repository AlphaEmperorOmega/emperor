from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_VERSION = 1


class WireCodecError(ValueError):
    """A value does not conform to the version 1 CLI wire protocol."""


def wire_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WireCodecError(f"{path} must be an object.")
    for key in value:
        if not isinstance(key, str):
            raise WireCodecError(f"{path} object keys must be strings.")
    return value


def wire_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise WireCodecError(f"{path} must be a list.")
    return value


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


def json_value_to_wire(value: Any, *, path: str = "$") -> Any:
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
        payload = wire_mapping(value, path)
        return {
            key: json_value_to_wire(item, path=f"{path}.{key}")
            for key, item in payload.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_value_to_wire(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise WireCodecError(f"{path} contains unsupported value {type(value).__name__}.")


def json_value_from_wire(value: object, *, path: str = "$") -> Any:
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
        payload = wire_mapping(value, path)
        return {
            key: json_value_from_wire(item, path=f"{path}.{key}")
            for key, item in payload.items()
        }
    if isinstance(value, list):
        return [
            json_value_from_wire(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise WireCodecError(f"{path} contains unsupported value {type(value).__name__}.")


def json_mapping_from_wire(value: object, *, path: str) -> dict[str, Any]:
    payload = wire_mapping(value, path)
    return {
        key: json_value_from_wire(item, path=f"{path}.{key}")
        for key, item in payload.items()
    }


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
