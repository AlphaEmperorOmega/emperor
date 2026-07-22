from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.cli._wire_packages import identity_from_wire, identity_to_wire
from model_runtime.cli._wire_shared import (
    json_mapping_from_wire,
    json_value_from_wire,
    json_value_to_wire,
    wire_bool,
    wire_fields,
    wire_int,
    wire_list,
    wire_literal,
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
    ConfigurationSchema,
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    InspectionResult,
    SearchAxis,
    SearchSpace,
)
from model_runtime.packages import serialize_config_value

_GRAPH_ROLES = {"architecture", "internal", "runtime"}


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


def configuration_schema_to_wire(schema: ConfigurationSchema) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(schema.identity),
        "fields": [
            {
                "key": field.key,
                "flag": field.flag,
                "section_path": list(field.section_path),
                "description": field.description,
                "value_type": field.value_type,
                "default": _scalar_to_wire(field.default, "$.fields[].default"),
                "nullable": field.nullable,
                "choices": [
                    _scalar_to_wire(choice, "$.fields[].choices[]")
                    for choice in field.choices
                ],
                "maximum": field.maximum,
                "locked": field.locked,
                "locked_value": _scalar_to_wire(
                    field.locked_value,
                    "$.fields[].locked_value",
                ),
                "locked_reason": field.locked_reason,
            }
            for field in schema.fields
        ],
    }
    configuration_schema_from_wire(payload)
    return payload


def configuration_schema_from_wire(payload: object) -> ConfigurationSchema:
    raw = wire_fields(
        payload,
        path="$",
        required=("identity", "fields"),
    )
    decoded_fields: list[ConfigurationField] = []
    for index, item in enumerate(wire_list(raw["fields"], "$.fields")):
        path = f"$.fields[{index}]"
        field = wire_fields(
            item,
            path=path,
            required=(
                "key",
                "flag",
                "section_path",
                "description",
                "value_type",
                "default",
                "nullable",
                "choices",
                "maximum",
                "locked",
                "locked_value",
                "locked_reason",
            ),
        )
        decoded_fields.append(
            ConfigurationField(
                key=wire_string(field["key"], f"{path}.key"),
                flag=wire_string(field["flag"], f"{path}.flag"),
                section_path=wire_string_list(
                    field["section_path"],
                    f"{path}.section_path",
                ),
                description=wire_string(
                    field["description"],
                    f"{path}.description",
                ),
                value_type=wire_string(
                    field["value_type"],
                    f"{path}.value_type",
                ),
                default=wire_scalar(field["default"], f"{path}.default"),
                nullable=wire_bool(field["nullable"], f"{path}.nullable"),
                choices=wire_scalar_list(field["choices"], f"{path}.choices"),
                maximum=wire_optional_number(
                    field["maximum"],
                    f"{path}.maximum",
                ),
                locked=wire_bool(field["locked"], f"{path}.locked"),
                locked_value=wire_scalar(
                    field["locked_value"],
                    f"{path}.locked_value",
                ),
                locked_reason=wire_string(
                    field["locked_reason"],
                    f"{path}.locked_reason",
                ),
            )
        )
    return ConfigurationSchema(
        identity=identity_from_wire(raw["identity"]),
        fields=tuple(decoded_fields),
    )


def search_space_to_wire(search_space: SearchSpace) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(search_space.identity),
        "preset": search_space.preset,
        "axes": [
            {
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
            for axis in search_space.axes
        ],
    }
    search_space_from_wire(payload)
    return payload


def search_space_from_wire(payload: object) -> SearchSpace:
    raw = wire_fields(
        payload,
        path="$",
        required=("identity", "preset", "axes"),
    )
    axes: list[SearchAxis] = []
    for index, item in enumerate(wire_list(raw["axes"], "$.axes")):
        path = f"$.axes[{index}]"
        axis = wire_fields(
            item,
            path=path,
            required=(
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
            ),
        )
        axes.append(
            SearchAxis(
                key=wire_string(axis["key"], f"{path}.key"),
                search_key=wire_string(
                    axis["search_key"],
                    f"{path}.search_key",
                ),
                section=wire_string(axis["section"], f"{path}.section"),
                value_type=wire_string(
                    axis["value_type"],
                    f"{path}.value_type",
                ),
                values=wire_scalar_list(axis["values"], f"{path}.values"),
                locked=wire_bool(axis["locked"], f"{path}.locked"),
                locked_value=wire_scalar(
                    axis["locked_value"],
                    f"{path}.locked_value",
                ),
                locked_reason=wire_string(
                    axis["locked_reason"],
                    f"{path}.locked_reason",
                ),
                locked_by_presets=wire_string_list(
                    axis["locked_by_presets"],
                    f"{path}.locked_by_presets",
                ),
                lock_reasons=wire_string_list(
                    axis["lock_reasons"],
                    f"{path}.lock_reasons",
                ),
            )
        )
    return SearchSpace(
        identity=identity_from_wire(raw["identity"]),
        preset=wire_optional_string(raw["preset"], "$.preset"),
        axes=tuple(axes),
    )


def _graph_configuration_to_wire(
    configuration: GraphConfiguration | None,
) -> dict[str, Any] | None:
    if configuration is None:
        return None
    return {
        "type_name": configuration.type_name,
        "fields": [
            {
                "key": field.key,
                "value": json_value_to_wire(field.value),
                "description": field.description,
            }
            for field in configuration.fields
        ],
    }


def inspection_result_to_wire(result: InspectionResult) -> dict[str, Any]:
    payload = {
        "identity": identity_to_wire(result.identity),
        "preset": result.preset,
        "parameter_count": wire_int(
            result.parameter_count,
            "$.parameter_count",
            minimum=0,
        ),
        "parameter_size_bytes": wire_int(
            result.parameter_size_bytes,
            "$.parameter_size_bytes",
            minimum=0,
        ),
        "nodes": [
            {
                "id": node.id,
                "type_name": node.type_name,
                "description": node.description,
                "path": node.path,
                "graph_role": wire_literal(
                    node.graph_role,
                    "$.nodes[].graph_role",
                    _GRAPH_ROLES,
                ),
                "parameter_count": wire_int(
                    node.parameter_count,
                    "$.nodes[].parameter_count",
                    minimum=0,
                ),
                "parameter_size_bytes": wire_int(
                    node.parameter_size_bytes,
                    "$.nodes[].parameter_size_bytes",
                    minimum=0,
                ),
                "details": json_value_to_wire(node.details),
                "configuration": _graph_configuration_to_wire(node.configuration),
            }
            for node in result.nodes
        ],
        "edges": [
            {"id": edge.id, "source": edge.source, "target": edge.target}
            for edge in result.edges
        ],
    }
    inspection_result_from_wire(payload)
    return payload


def _graph_configuration_from_wire(
    value: object,
    *,
    path: str,
) -> GraphConfiguration | None:
    if value is None:
        return None
    raw = wire_fields(
        value,
        path=path,
        required=("type_name", "fields"),
    )
    fields: list[GraphConfigurationField] = []
    for index, item in enumerate(wire_list(raw["fields"], f"{path}.fields")):
        field_path = f"{path}.fields[{index}]"
        field = wire_fields(
            item,
            path=field_path,
            required=("key", "value", "description"),
        )
        fields.append(
            GraphConfigurationField(
                key=wire_string(field["key"], f"{field_path}.key"),
                value=json_value_from_wire(
                    field["value"],
                    path=f"{field_path}.value",
                ),
                description=wire_optional_string(
                    field["description"],
                    f"{field_path}.description",
                ),
            )
        )
    return GraphConfiguration(
        type_name=wire_string(raw["type_name"], f"{path}.type_name"),
        fields=tuple(fields),
    )


def inspection_result_from_wire(payload: object) -> InspectionResult:
    raw = wire_fields(
        payload,
        path="$",
        required=(
            "identity",
            "preset",
            "parameter_count",
            "parameter_size_bytes",
            "nodes",
            "edges",
        ),
    )
    nodes: list[GraphNode] = []
    for index, item in enumerate(wire_list(raw["nodes"], "$.nodes")):
        path = f"$.nodes[{index}]"
        node = wire_fields(
            item,
            path=path,
            required=(
                "id",
                "type_name",
                "description",
                "path",
                "graph_role",
                "parameter_count",
                "parameter_size_bytes",
                "details",
                "configuration",
            ),
        )
        nodes.append(
            GraphNode(
                id=wire_string(node["id"], f"{path}.id"),
                type_name=wire_string(node["type_name"], f"{path}.type_name"),
                description=wire_optional_string(
                    node["description"],
                    f"{path}.description",
                ),
                path=wire_string(node["path"], f"{path}.path"),
                graph_role=wire_literal(
                    node["graph_role"],
                    f"{path}.graph_role",
                    _GRAPH_ROLES,
                ),
                parameter_count=wire_int(
                    node["parameter_count"],
                    f"{path}.parameter_count",
                    minimum=0,
                ),
                parameter_size_bytes=wire_int(
                    node["parameter_size_bytes"],
                    f"{path}.parameter_size_bytes",
                    minimum=0,
                ),
                details=json_mapping_from_wire(
                    node["details"],
                    path=f"{path}.details",
                ),
                configuration=_graph_configuration_from_wire(
                    node["configuration"],
                    path=f"{path}.configuration",
                ),
            )
        )

    edges: list[GraphEdge] = []
    for index, item in enumerate(wire_list(raw["edges"], "$.edges")):
        path = f"$.edges[{index}]"
        edge = wire_fields(
            item,
            path=path,
            required=("id", "source", "target"),
        )
        edges.append(
            GraphEdge(
                id=wire_string(edge["id"], f"{path}.id"),
                source=wire_string(edge["source"], f"{path}.source"),
                target=wire_string(edge["target"], f"{path}.target"),
            )
        )

    return InspectionResult(
        identity=identity_from_wire(raw["identity"]),
        preset=wire_string(raw["preset"], "$.preset"),
        parameter_count=wire_int(
            raw["parameter_count"],
            "$.parameter_count",
            minimum=0,
        ),
        parameter_size_bytes=wire_int(
            raw["parameter_size_bytes"],
            "$.parameter_size_bytes",
            minimum=0,
        ),
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


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
