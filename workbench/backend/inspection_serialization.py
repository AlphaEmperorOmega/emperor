"""Workbench representation mapping for Emperor Inspection records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from emperor.inspection import (
    ConfigurationSchema,
    GraphConfiguration,
    GraphEdge,
    GraphNode,
    InspectionResult,
    ModelGraph,
    SearchSpace,
)
from emperor.model_packages import (
    ModelPackage,
    dataset_label,
    dataset_name,
)


def _camel_case(key: str) -> str:
    pieces = key.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


def _workbench_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            _camel_case(str(key)): _workbench_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_workbench_value(item) for item in value]
    return value


def _configuration_payload(
    configuration: GraphConfiguration | None,
) -> dict[str, Any] | None:
    if configuration is None:
        return None
    fields = []
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
            "details": _workbench_value(node.details),
            "config": _configuration_payload(node.configuration),
        }
    )
    return payload


def graph_edge_payload(edge: GraphEdge) -> dict[str, str]:
    return {"id": edge.id, "source": edge.source, "target": edge.target}


def model_graph_payload(
    graph: ModelGraph,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return (
        [graph_node_payload(node) for node in graph.nodes],
        [graph_edge_payload(edge) for edge in graph.edges],
    )


def inspection_result_payload(result: InspectionResult) -> dict[str, Any]:
    return {
        **result.identity.to_payload(),
        "preset": result.preset,
        "parameterCount": result.parameter_count,
        "parameterSizeBytes": result.parameter_size_bytes,
        "nodes": [graph_node_payload(node) for node in result.nodes],
        "edges": [graph_edge_payload(edge) for edge in result.edges],
    }


def configuration_schema_payload(schema: ConfigurationSchema) -> dict[str, Any]:
    fields = []
    for field in schema.fields:
        fields.append(
            {
                "key": field.key,
                "configKey": field.key,
                "flag": field.flag,
                "label": field.key.lower().replace("_", " "),
                "section": field.section_path[-1],
                "sectionPath": list(field.section_path),
                "description": field.description,
                "type": field.value_type,
                "default": field.default,
                "nullable": field.nullable,
                "choices": list(field.choices),
                "maximum": field.maximum,
                "locked": field.locked,
                "lockedValue": field.locked_value,
                "lockedReason": field.locked_reason,
            }
        )
    return {**schema.identity.to_payload(), "fields": fields}


def search_space_payload(search_space: SearchSpace) -> dict[str, Any]:
    axes = []
    for axis in search_space.axes:
        axes.append(
            {
                "key": axis.key,
                "configKey": axis.key,
                "searchKey": axis.search_key,
                "label": axis.key.lower().replace("_", " "),
                "section": axis.section,
                "type": axis.value_type,
                "values": list(axis.values),
                "locked": axis.locked,
                "lockedValue": axis.locked_value,
                "lockedReason": axis.locked_reason,
                "lockedByPresets": list(axis.locked_by_presets),
                "lockReasons": list(axis.lock_reasons),
            }
        )
    return {
        **search_space.identity.to_payload(),
        "preset": search_space.preset,
        "axes": axes,
    }


def model_presets_payload(package: ModelPackage) -> list[dict[str, str]]:
    return [
        {
            "name": package.preset_name(preset),
            "label": preset.name,
            "description": package.preset_description(preset),
        }
        for preset in package.preset_type
    ]


def _dataset_payload(dataset: type) -> dict[str, Any]:
    return {
        "name": dataset_name(dataset),
        "label": dataset_label(dataset),
        "inputDim": int(getattr(dataset, "flattened_input_dim", 0) or 0),
        "outputDim": int(getattr(dataset, "num_classes", 0) or 0),
    }


def model_datasets_payload(package: ModelPackage) -> dict[str, Any]:
    return {
        "defaultExperimentTask": package.task_name(package.default_experiment_task),
        "datasetGroups": [
            {
                "experimentTask": package.task_name(task),
                "label": package.task_label(task),
                "datasets": [_dataset_payload(dataset) for dataset in datasets],
            }
            for task, datasets in package.dataset_metadata.items()
        ],
    }


def model_monitors_payload(package: ModelPackage) -> list[dict[str, Any]]:
    return [
        {
            "name": option.name,
            "label": option.label,
            "description": option.description,
            "kinds": list(option.kinds),
            "defaultEnabled": option.default_enabled,
        }
        for option in package.monitor_options()
    ]


__all__ = [
    "configuration_schema_payload",
    "graph_edge_payload",
    "graph_node_payload",
    "inspection_result_payload",
    "model_datasets_payload",
    "model_graph_payload",
    "model_monitors_payload",
    "model_presets_payload",
    "search_space_payload",
]
