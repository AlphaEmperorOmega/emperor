from __future__ import annotations

from typing import Any


def pop_updates(
    values: dict[str, Any],
    field_map: dict[str, str],
) -> dict[str, Any]:
    """Remove mapped flat values and return their structured field names."""

    return {
        structured_field: values.pop(flat_field)
        for flat_field, structured_field in field_map.items()
        if flat_field in values
    }
