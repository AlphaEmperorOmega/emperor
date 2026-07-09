from __future__ import annotations

from typing import Any, TypeAlias, cast

SerializedConfigValue: TypeAlias = bool | int | float | str | None


def serialize_config_value(value: Any) -> SerializedConfigValue:
    if hasattr(value, "name"):
        return cast(SerializedConfigValue, value.name)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
