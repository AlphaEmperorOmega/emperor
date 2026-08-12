from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_value_from_wire,
    json_value_to_wire,
    wire_bool,
    wire_fields,
    wire_list,
    wire_literal,
    wire_optional_int,
    wire_string,
)
from model_runtime.runs import PlanningBudget, SearchAxisSelection, SearchSpec
from model_runtime.runs.records import SearchMode

_SEARCH_MODES = {"grid", "random"}
_DEFAULT_BUDGET = PlanningBudget()
_MAX_WIRE_SEARCH_AXES = _DEFAULT_BUDGET.max_axes or 16
_MAX_WIRE_SEARCH_VALUES = _DEFAULT_BUDGET.max_values_per_axis or 50


def _require_sequence_limit(
    values: Sequence[Any],
    path: str,
    maximum_items: int,
) -> None:
    if len(values) > maximum_items:
        raise WireCodecError(f"{path} must contain at most {maximum_items} items.")


def search_spec_to_wire_at(
    search: SearchSpec | None,
    path: str,
) -> dict[str, Any] | None:
    if search is None:
        return None
    if search.axes is not None:
        _require_sequence_limit(
            search.axes,
            f"{path}.axes",
            _MAX_WIRE_SEARCH_AXES,
        )
        for index, axis in enumerate(search.axes):
            if axis.values is not None:
                _require_sequence_limit(
                    axis.values,
                    f"{path}.axes[{index}].values",
                    _MAX_WIRE_SEARCH_VALUES,
                )
    payload = {
        "mode": wire_literal(search.mode, f"{path}.mode", _SEARCH_MODES),
        "axes": (
            None
            if search.axes is None
            else [
                {
                    "key": axis.key,
                    "values": (
                        None
                        if axis.values is None
                        else [
                            json_value_to_wire(
                                value,
                                path=f"{path}.axes[].values[]",
                            )
                            for value in axis.values
                        ]
                    ),
                    "allow_custom_values": axis.allow_custom_values,
                }
                for axis in search.axes
            ]
        ),
        "random_samples": search.random_samples,
    }
    search_spec_from_wire_at(payload, path)
    return payload


def search_spec_to_wire(search: SearchSpec | None) -> dict[str, Any] | None:
    return search_spec_to_wire_at(search, "$.search")


def _search_axis_from_wire(
    item: object,
    index: int,
    search_path: str,
) -> SearchAxisSelection:
    path = f"{search_path}.axes[{index}]"
    axis = wire_fields(
        item,
        path=path,
        required=("key",),
        optional=("values", "allow_custom_values"),
    )
    raw_values = axis.get("values")
    return SearchAxisSelection(
        key=wire_string(axis["key"], f"{path}.key"),
        values=(
            None
            if raw_values is None
            else tuple(
                json_value_from_wire(
                    value,
                    path=f"{path}.values[{value_index}]",
                )
                for value_index, value in enumerate(
                    wire_list(
                        raw_values,
                        f"{path}.values",
                        maximum_items=_MAX_WIRE_SEARCH_VALUES,
                    )
                )
            )
        ),
        allow_custom_values=wire_bool(
            axis.get("allow_custom_values", False),
            f"{path}.allow_custom_values",
        ),
    )


def _search_axes_from_wire(
    payload: object,
    path: str,
) -> tuple[SearchAxisSelection, ...] | None:
    if payload is None:
        return None
    decoded_axes: list[SearchAxisSelection] = []
    for index, item in enumerate(
        wire_list(
            payload,
            f"{path}.axes",
            maximum_items=_MAX_WIRE_SEARCH_AXES,
        )
    ):
        decoded_axes.append(_search_axis_from_wire(item, index, path))
    return tuple(decoded_axes)


def search_spec_from_wire_at(payload: object, path: str) -> SearchSpec | None:
    if payload is None:
        return None
    raw = wire_fields(
        payload,
        path=path,
        required=("mode",),
        optional=("axes", "random_samples"),
    )
    axes = _search_axes_from_wire(raw.get("axes"), path)
    return SearchSpec(
        mode=cast(
            SearchMode,
            wire_literal(raw["mode"], f"{path}.mode", _SEARCH_MODES),
        ),
        axes=axes,
        random_samples=wire_optional_int(
            raw.get("random_samples"),
            f"{path}.random_samples",
            minimum=1,
        ),
    )


def search_spec_from_wire(payload: object) -> SearchSpec | None:
    return search_spec_from_wire_at(payload, "$.search")


__all__ = [
    "search_spec_from_wire",
    "search_spec_from_wire_at",
    "search_spec_to_wire",
    "search_spec_to_wire_at",
]
