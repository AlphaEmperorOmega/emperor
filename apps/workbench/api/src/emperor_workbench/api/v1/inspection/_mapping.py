from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import (
    GraphConfiguration,
    GraphEdge,
    GraphNode,
    InspectionResult,
)

from emperor_workbench.api.v1.inspection._contracts import InspectResponse


def _camel_case(key: str) -> str:
    pieces = key.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _http_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_camel_case(str(key)): _http_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_http_value(item) for item in value]
    return value


def _configuration_payload(
    configuration: GraphConfiguration | None,
) -> dict[str, Any] | None:
    if configuration is None:
        return None
    fields: list[dict[str, Any]] = []
    for field in configuration.fields:
        payload = {"key": field.key, "value": field.value}
        if field.description is not None:
            payload["description"] = field.description
        fields.append(payload)
    return {"typeName": configuration.type_name, "fields": fields}


def graph_node_payload(node: GraphNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": node.id,
        "label": node.type_name,
        "typeName": node.type_name,
    }
    if node.description is not None:
        payload["description"] = node.description
    payload.update(
        {
            "path": node.path,
            "graphRole": node.graph_role,
            "parameterCount": node.parameter_count,
            "parameterSizeBytes": node.parameter_size_bytes,
            "details": _http_value(node.details),
            "config": _configuration_payload(node.configuration),
        }
    )
    return payload


def graph_edge_payload(edge: GraphEdge) -> dict[str, str]:
    return {"id": edge.id, "source": edge.source, "target": edge.target}


def inspection_response(result: InspectionResult) -> InspectResponse:
    return InspectResponse.model_validate(
        {
            **result.identity.to_payload(),
            "preset": result.preset,
            "parameterCount": result.parameter_count,
            "parameterSizeBytes": result.parameter_size_bytes,
            "nodes": [graph_node_payload(node) for node in result.nodes],
            "edges": [graph_edge_payload(edge) for edge in result.edges],
        }
    )


__all__ = ["inspection_response"]
