from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, cast

from model_runtime.cli._wire_packages import identity_from_wire, identity_to_wire
from model_runtime.cli._wire_shared import (
    WireCodecError,
    json_mapping_from_wire,
    json_value_from_wire,
    json_value_to_wire,
    require_sequence_limit,
    wire_fields,
    wire_int,
    wire_list,
    wire_literal,
    wire_optional_string,
    wire_string,
)
from model_runtime.inspection.capture_limits import estimated_output_bytes
from model_runtime.inspection.records import (
    GraphConfiguration,
    GraphConfigurationField,
    GraphEdge,
    GraphNode,
    GraphRole,
    InspectionResult,
)

_GRAPH_ROLES = {"architecture", "internal", "runtime"}
_INSPECTION_RESULT_FIELDS = (
    "identity",
    "preset",
    "parameter_count",
    "parameter_size_bytes",
    "nodes",
    "edges",
)
_GRAPH_NODE_FIELDS = (
    "id",
    "type_name",
    "description",
    "path",
    "graph_role",
    "parameter_count",
    "parameter_size_bytes",
    "details",
    "configuration",
)


@dataclass(frozen=True, slots=True)
class InspectionWireLimits:
    maximum_graph_nodes: int = 8_192
    maximum_graph_edges: int = 8_192
    maximum_configuration_fields: int = 512
    maximum_output_bytes: int = 16 * 1024**2


def _require_output_budget(
    value: object,
    limits: InspectionWireLimits,
) -> None:
    maximum = limits.maximum_output_bytes
    if estimated_output_bytes(value, maximum) > maximum:
        raise WireCodecError(f"Inspection output byte limit of {maximum} exceeded.")


def _graph_configuration_to_wire(
    configuration: GraphConfiguration | None,
    limits: InspectionWireLimits,
) -> dict[str, Any] | None:
    if configuration is None:
        return None
    require_sequence_limit(
        configuration.fields,
        "$.nodes[].configuration.fields",
        limits.maximum_configuration_fields,
    )
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


def _graph_node_to_wire(
    node: GraphNode,
    limits: InspectionWireLimits,
) -> dict[str, Any]:
    return {
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
        "configuration": _graph_configuration_to_wire(node.configuration, limits),
    }


def _graph_edge_to_wire(edge: GraphEdge) -> dict[str, str]:
    return {"id": edge.id, "source": edge.source, "target": edge.target}


def encode_inspection_result(
    result: InspectionResult,
    *,
    limits: InspectionWireLimits,
) -> dict[str, Any]:
    require_sequence_limit(
        result.nodes,
        "$.nodes",
        limits.maximum_graph_nodes,
    )
    require_sequence_limit(
        result.edges,
        "$.edges",
        limits.maximum_graph_edges,
    )
    _require_output_budget(result, limits)
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
        "nodes": [_graph_node_to_wire(node, limits) for node in result.nodes],
        "edges": [_graph_edge_to_wire(edge) for edge in result.edges],
    }
    decode_inspection_result(payload, limits=limits)
    return payload


def _graph_configuration_from_wire(
    value: object,
    *,
    path: str,
    limits: InspectionWireLimits,
) -> GraphConfiguration | None:
    if value is None:
        return None
    raw = wire_fields(
        value,
        path=path,
        required=("type_name", "fields"),
    )
    fields = tuple(
        _graph_configuration_field_from_wire(item, path, index)
        for index, item in enumerate(
            wire_list(
                raw["fields"],
                f"{path}.fields",
                maximum_items=limits.maximum_configuration_fields,
            )
        )
    )
    return GraphConfiguration(
        type_name=wire_string(raw["type_name"], f"{path}.type_name"),
        fields=fields,
    )


def _graph_configuration_field_from_wire(
    item: object,
    configuration_path: str,
    index: int,
) -> GraphConfigurationField:
    path = f"{configuration_path}.fields[{index}]"
    field = wire_fields(
        item,
        path=path,
        required=("key", "value", "description"),
    )
    return GraphConfigurationField(
        key=wire_string(field["key"], f"{path}.key"),
        value=json_value_from_wire(field["value"], path=f"{path}.value"),
        description=wire_optional_string(
            field["description"],
            f"{path}.description",
        ),
    )


def _graph_role(value: object, path: str) -> GraphRole:
    return cast(GraphRole, wire_literal(value, path, _GRAPH_ROLES))


class _InspectionResultDecoder:
    def __init__(
        self,
        payload: object,
        limits: InspectionWireLimits,
    ) -> None:
        self._raw = wire_fields(
            payload,
            path="$",
            required=_INSPECTION_RESULT_FIELDS,
        )
        self._limits = limits
        self._nodes: list[GraphNode] = []
        self._node_ids: set[str] = set()
        self._edges: list[GraphEdge] = []
        self._edge_ids: set[str] = set()

    def decode(self) -> InspectionResult:
        self._decode_nodes()
        self._decode_edges()
        self._require_acyclic_graph()
        return InspectionResult(
            identity=identity_from_wire(self._raw["identity"]),
            preset=wire_string(self._raw["preset"], "$.preset"),
            parameter_count=wire_int(
                self._raw["parameter_count"],
                "$.parameter_count",
                minimum=0,
            ),
            parameter_size_bytes=wire_int(
                self._raw["parameter_size_bytes"],
                "$.parameter_size_bytes",
                minimum=0,
            ),
            nodes=tuple(self._nodes),
            edges=tuple(self._edges),
        )

    def _decode_nodes(self) -> None:
        items = wire_list(
            self._raw["nodes"],
            "$.nodes",
            maximum_items=self._limits.maximum_graph_nodes,
        )
        for index, item in enumerate(items):
            path = f"$.nodes[{index}]"
            decoded_node = self._decode_node(item, path)
            if decoded_node.id in self._node_ids:
                raise WireCodecError(
                    f"{path}.id contains duplicate graph node id {decoded_node.id!r}."
                )
            self._node_ids.add(decoded_node.id)
            self._nodes.append(decoded_node)

    def _decode_node(self, item: object, path: str) -> GraphNode:
        node = wire_fields(item, path=path, required=_GRAPH_NODE_FIELDS)
        return GraphNode(
            id=wire_string(node["id"], f"{path}.id"),
            type_name=wire_string(node["type_name"], f"{path}.type_name"),
            description=wire_optional_string(
                node["description"],
                f"{path}.description",
            ),
            path=wire_string(node["path"], f"{path}.path"),
            graph_role=_graph_role(node["graph_role"], f"{path}.graph_role"),
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
                limits=self._limits,
            ),
        )

    def _decode_edges(self) -> None:
        items = wire_list(
            self._raw["edges"],
            "$.edges",
            maximum_items=self._limits.maximum_graph_edges,
        )
        for index, item in enumerate(items):
            path = f"$.edges[{index}]"
            decoded_edge = self._decode_edge(item, path)
            self._validate_edge(decoded_edge, path)
            self._edge_ids.add(decoded_edge.id)
            self._edges.append(decoded_edge)

    @staticmethod
    def _decode_edge(item: object, path: str) -> GraphEdge:
        edge = wire_fields(
            item,
            path=path,
            required=("id", "source", "target"),
        )
        return GraphEdge(
            id=wire_string(edge["id"], f"{path}.id"),
            source=wire_string(edge["source"], f"{path}.source"),
            target=wire_string(edge["target"], f"{path}.target"),
        )

    def _validate_edge(self, edge: GraphEdge, path: str) -> None:
        if edge.id in self._edge_ids:
            raise WireCodecError(
                f"{path}.id contains duplicate graph edge id {edge.id!r}."
            )
        for endpoint_name, endpoint in (
            ("source", edge.source),
            ("target", edge.target),
        ):
            if endpoint not in self._node_ids:
                raise WireCodecError(
                    f"{path}.{endpoint_name} references unknown graph node "
                    f"{endpoint!r}."
                )

    def _require_acyclic_graph(self) -> None:
        indegree = dict.fromkeys(self._node_ids, 0)
        targets_by_source: dict[str, list[str]] = {
            node_id: [] for node_id in self._node_ids
        }
        for edge in self._edges:
            indegree[edge.target] += 1
            targets_by_source[edge.source].append(edge.target)
        ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited_node_count = 0
        while ready:
            source = ready.popleft()
            visited_node_count += 1
            for target in targets_by_source[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited_node_count != len(self._node_ids):
            raise WireCodecError("$.edges graph must be acyclic.")


def decode_inspection_result(
    payload: object,
    *,
    limits: InspectionWireLimits,
) -> InspectionResult:
    _require_output_budget(payload, limits)
    return _InspectionResultDecoder(payload, limits).decode()


__all__ = [
    "InspectionWireLimits",
    "decode_inspection_result",
    "encode_inspection_result",
]
