from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.cli._wire_graph import (
    InspectionWireLimits,
    decode_inspection_result,
    encode_inspection_result,
)
from model_runtime.cli._wire_packages import identity_from_wire, identity_to_wire
from model_runtime.cli._wire_shared import (
    wire_bool,
    wire_fields,
    wire_list,
    wire_mapping,
    wire_optional_number,
    wire_optional_string,
    wire_scalar,
    wire_scalar_list,
    wire_string,
    wire_string_list,
)
from model_runtime.inspection import (
    ConfigurationField,
    ConfigurationFieldCondition,
    ConfigurationSchema,
    InspectionResult,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import serialize_config_value

_INSPECTION_WIRE_LIMITS = InspectionWireLimits()
_CONFIGURATION_FIELD_FIELDS = (
    "key",
    "flag",
    "section_path",
    "description",
    "value_type",
    "default",
    "nullable",
    "choices",
    "applicableWhen",
    "maximum",
    "locked",
    "locked_value",
    "locked_reason",
)
_SEARCH_AXIS_FIELDS = (
    "key",
    "search_key",
    "section",
    "value_type",
    "values",
    "locked",
    "locked_value",
    "locked_reason",
    "locked_by_presets",
    "lock_reasons",
)


def _scalar_to_wire(value: object, path: str) -> object:
    return wire_scalar(value, path)


def configuration_values_to_wire(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = wire_mapping(values, "$.configuration_values")
    return {
        key: wire_scalar(
            serialize_config_value(value),
            f"$.configuration_values.{key}",
        )
        for key, value in payload.items()
    }


def preset_locks_to_wire(locks: Mapping[str, Any]) -> dict[str, Any]:
    payload = wire_mapping(locks, "$.preset_locks")
    encoded: dict[str, Any] = {}
    for key, lock in payload.items():
        reason = wire_string(
            getattr(lock, "reason", None),
            f"$.preset_locks.{key}.reason",
        )
        encoded[key] = {
            "value": wire_scalar(
                serialize_config_value(getattr(lock, "value", None)),
                f"$.preset_locks.{key}.value",
            ),
            "reason": reason,
        }
    return encoded


def _configuration_condition_to_wire(
    condition: ConfigurationFieldCondition,
) -> dict[str, Any]:
    return {
        "key": condition.key,
        "values": [
            _scalar_to_wire(value, "$.fields[].applicableWhen[].values[]")
            for value in condition.values
        ],
    }


def _configuration_field_to_wire(field: ConfigurationField) -> dict[str, Any]:
    return {
        "key": field.key,
        "flag": field.flag,
        "section_path": list(field.section_path),
        "description": field.description,
        "value_type": field.value_type,
        "default": _scalar_to_wire(field.default, "$.fields[].default"),
        "nullable": field.nullable,
        "choices": [
            _scalar_to_wire(choice, "$.fields[].choices[]") for choice in field.choices
        ],
        "applicableWhen": [
            _configuration_condition_to_wire(condition)
            for condition in field.applicable_when
        ],
        "maximum": field.maximum,
        "locked": field.locked,
        "locked_value": _scalar_to_wire(
            field.locked_value,
            "$.fields[].locked_value",
        ),
        "locked_reason": field.locked_reason,
    }


def configuration_schema_to_wire(schema: ConfigurationSchema) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(schema.identity),
        "fields": [_configuration_field_to_wire(field) for field in schema.fields],
    }
    configuration_schema_from_wire(payload)
    return payload


def _configuration_condition_from_wire(
    item: object,
    field_path: str,
    index: int,
) -> ConfigurationFieldCondition:
    path = f"{field_path}.applicableWhen[{index}]"
    condition = wire_fields(
        item,
        path=path,
        required=("key", "values"),
    )
    return ConfigurationFieldCondition(
        key=wire_string(condition["key"], f"{path}.key"),
        values=wire_scalar_list(condition["values"], f"{path}.values"),
    )


def _configuration_conditions_from_wire(
    value: object,
    field_path: str,
) -> tuple[ConfigurationFieldCondition, ...]:
    return tuple(
        _configuration_condition_from_wire(item, field_path, index)
        for index, item in enumerate(wire_list(value, f"{field_path}.applicableWhen"))
    )


def _configuration_field_from_wire(
    item: object,
    index: int,
) -> ConfigurationField:
    path = f"$.fields[{index}]"
    field = wire_fields(item, path=path, required=_CONFIGURATION_FIELD_FIELDS)
    applicable_when = _configuration_conditions_from_wire(
        field["applicableWhen"],
        path,
    )
    return ConfigurationField(
        key=wire_string(field["key"], f"{path}.key"),
        flag=wire_string(field["flag"], f"{path}.flag"),
        section_path=wire_string_list(field["section_path"], f"{path}.section_path"),
        description=wire_string(field["description"], f"{path}.description"),
        value_type=wire_string(field["value_type"], f"{path}.value_type"),
        default=wire_scalar(field["default"], f"{path}.default"),
        nullable=wire_bool(field["nullable"], f"{path}.nullable"),
        choices=wire_scalar_list(field["choices"], f"{path}.choices"),
        applicable_when=applicable_when,
        maximum=wire_optional_number(field["maximum"], f"{path}.maximum"),
        locked=wire_bool(field["locked"], f"{path}.locked"),
        locked_value=wire_scalar(field["locked_value"], f"{path}.locked_value"),
        locked_reason=wire_string(field["locked_reason"], f"{path}.locked_reason"),
    )


def configuration_schema_from_wire(payload: object) -> ConfigurationSchema:
    raw = wire_fields(
        payload,
        path="$",
        required=("identity", "fields"),
    )
    decoded_fields = tuple(
        _configuration_field_from_wire(item, index)
        for index, item in enumerate(wire_list(raw["fields"], "$.fields"))
    )
    return ConfigurationSchema(
        identity=identity_from_wire(raw["identity"]),
        fields=decoded_fields,
    )


def _search_axis_to_wire(axis: SearchAxis) -> dict[str, Any]:
    return {
        "key": axis.key,
        "search_key": axis.search_key,
        "section": axis.section,
        "value_type": axis.value_type,
        "values": [
            _scalar_to_wire(value, "$.axes[].values[]") for value in axis.values
        ],
        "locked": axis.locked,
        "locked_value": _scalar_to_wire(
            axis.locked_value,
            "$.axes[].locked_value",
        ),
        "locked_reason": axis.locked_reason,
        "locked_by_presets": list(axis.locked_by_presets),
        "lock_reasons": list(axis.lock_reasons),
    }


def search_space_to_wire(search_space: SearchSpace) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(search_space.identity),
        "preset": search_space.preset,
        "axes": [_search_axis_to_wire(axis) for axis in search_space.axes],
    }
    search_space_from_wire(payload)
    return payload


def _search_axis_from_wire(item: object, index: int) -> SearchAxis:
    path = f"$.axes[{index}]"
    axis = wire_fields(item, path=path, required=_SEARCH_AXIS_FIELDS)
    return SearchAxis(
        key=wire_string(axis["key"], f"{path}.key"),
        search_key=wire_string(axis["search_key"], f"{path}.search_key"),
        section=wire_string(axis["section"], f"{path}.section"),
        value_type=wire_string(axis["value_type"], f"{path}.value_type"),
        values=wire_scalar_list(axis["values"], f"{path}.values"),
        locked=wire_bool(axis["locked"], f"{path}.locked"),
        locked_value=wire_scalar(axis["locked_value"], f"{path}.locked_value"),
        locked_reason=wire_string(axis["locked_reason"], f"{path}.locked_reason"),
        locked_by_presets=wire_string_list(
            axis["locked_by_presets"],
            f"{path}.locked_by_presets",
        ),
        lock_reasons=wire_string_list(
            axis["lock_reasons"],
            f"{path}.lock_reasons",
        ),
    )


def search_space_from_wire(payload: object) -> SearchSpace:
    raw = wire_fields(
        payload,
        path="$",
        required=("identity", "preset", "axes"),
    )
    axes = tuple(
        _search_axis_from_wire(item, index)
        for index, item in enumerate(wire_list(raw["axes"], "$.axes"))
    )
    return SearchSpace(
        identity=identity_from_wire(raw["identity"]),
        preset=wire_optional_string(raw["preset"], "$.preset"),
        axes=axes,
    )


def inspection_result_to_wire(result: InspectionResult) -> dict[str, Any]:
    return encode_inspection_result(result, limits=_INSPECTION_WIRE_LIMITS)


def inspection_result_from_wire(payload: object) -> InspectionResult:
    return decode_inspection_result(payload, limits=_INSPECTION_WIRE_LIMITS)


__all__ = [
    "configuration_values_to_wire",
    "configuration_schema_from_wire",
    "configuration_schema_to_wire",
    "inspection_result_from_wire",
    "inspection_result_to_wire",
    "preset_locks_to_wire",
    "search_space_from_wire",
    "search_space_to_wire",
]
