from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

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
from model_runtime.packages import (
    ModelIdentity,
    ModelPackage,
    dataset_label,
    dataset_name,
)
from model_runtime.runs import (
    PlanningBudget,
    RunParameter,
    RunPlan,
    RunRequest,
    RunSpec,
    SearchAxisSelection,
    SearchSpec,
    SubmittedRun,
)

PROTOCOL_VERSION = 1


def to_wire(value: Any) -> Any:
    """Recursively project runtime values onto finite JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_wire(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [to_wire(item) for item in value]
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def identity_from_wire(payload: Mapping[str, Any]) -> ModelIdentity:
    return ModelIdentity(str(payload["model_type"]), str(payload["model"]))


def configuration_schema_from_wire(payload: Mapping[str, Any]) -> ConfigurationSchema:
    return ConfigurationSchema(
        identity=identity_from_wire(_mapping(payload["identity"])),
        fields=tuple(
            ConfigurationField(
                key=str(field["key"]),
                flag=str(field["flag"]),
                section_path=tuple(str(item) for item in field["section_path"]),
                description=str(field["description"]),
                value_type=str(field["value_type"]),
                default=field.get("default"),
                nullable=bool(field["nullable"]),
                choices=tuple(field.get("choices") or ()),
                maximum=field.get("maximum"),
                locked=bool(field.get("locked", False)),
                locked_value=field.get("locked_value"),
                locked_reason=str(field.get("locked_reason") or ""),
            )
            for field in _mappings(payload["fields"])
        ),
    )


def search_space_from_wire(payload: Mapping[str, Any]) -> SearchSpace:
    return SearchSpace(
        identity=identity_from_wire(_mapping(payload["identity"])),
        preset=str(payload["preset"]) if payload.get("preset") is not None else None,
        axes=tuple(
            SearchAxis(
                key=str(axis["key"]),
                search_key=str(axis["search_key"]),
                section=str(axis["section"]),
                value_type=str(axis["value_type"]),
                values=tuple(axis.get("values") or ()),
                locked=bool(axis.get("locked", False)),
                locked_value=axis.get("locked_value"),
                locked_reason=str(axis.get("locked_reason") or ""),
                locked_by_presets=tuple(
                    str(item) for item in axis.get("locked_by_presets") or ()
                ),
                lock_reasons=tuple(
                    str(item) for item in axis.get("lock_reasons") or ()
                ),
            )
            for axis in _mappings(payload["axes"])
        ),
    )


def inspection_result_from_wire(payload: Mapping[str, Any]) -> InspectionResult:
    nodes: list[GraphNode] = []
    for raw_node in _mappings(payload["nodes"]):
        raw_configuration = raw_node.get("configuration")
        configuration = None
        if isinstance(raw_configuration, Mapping):
            configuration = GraphConfiguration(
                type_name=str(raw_configuration["type_name"]),
                fields=tuple(
                    GraphConfigurationField(
                        key=str(field["key"]),
                        value=field.get("value"),
                        description=(
                            str(field["description"])
                            if field.get("description") is not None
                            else None
                        ),
                    )
                    for field in _mappings(raw_configuration["fields"])
                ),
            )
        nodes.append(
            GraphNode(
                id=str(raw_node["id"]),
                type_name=str(raw_node["type_name"]),
                description=(
                    str(raw_node["description"])
                    if raw_node.get("description") is not None
                    else None
                ),
                path=str(raw_node["path"]),
                graph_role=raw_node["graph_role"],
                parameter_count=int(raw_node["parameter_count"]),
                parameter_size_bytes=int(raw_node["parameter_size_bytes"]),
                details=dict(_mapping(raw_node.get("details") or {})),
                configuration=configuration,
            )
        )
    return InspectionResult(
        identity=identity_from_wire(_mapping(payload["identity"])),
        preset=str(payload["preset"]),
        parameter_count=int(payload["parameter_count"]),
        parameter_size_bytes=int(payload["parameter_size_bytes"]),
        nodes=tuple(nodes),
        edges=tuple(
            GraphEdge(
                id=str(edge["id"]),
                source=str(edge["source"]),
                target=str(edge["target"]),
            )
            for edge in _mappings(payload["edges"])
        ),
    )


def search_spec_from_wire(payload: object) -> SearchSpec | None:
    if payload is None:
        return None
    raw = _mapping(payload)
    raw_axes = raw.get("axes")
    return SearchSpec(
        mode=raw["mode"],
        axes=(
            None
            if raw_axes is None
            else tuple(
                SearchAxisSelection(
                    key=str(axis["key"]),
                    values=(
                        tuple(axis["values"])
                        if axis.get("values") is not None
                        else None
                    ),
                    allow_custom_values=bool(axis.get("allow_custom_values", False)),
                )
                for axis in _mappings(raw_axes)
            )
        ),
        random_samples=(
            int(raw["random_samples"])
            if raw.get("random_samples") is not None
            else None
        ),
    )


def run_request_from_wire(payload: Mapping[str, Any]) -> RunRequest:
    return RunRequest(
        presets=tuple(str(item) for item in payload["presets"]),
        datasets=tuple(str(item) for item in payload["datasets"]),
        experiment_task=(
            str(payload["experiment_task"])
            if payload.get("experiment_task") is not None
            else None
        ),
        overrides=dict(_mapping(payload.get("overrides") or {})),
        search=search_spec_from_wire(payload.get("search")),
    )


def planning_budget_from_wire(payload: Mapping[str, Any]) -> PlanningBudget:
    return PlanningBudget(
        max_axes=_optional_int(payload.get("max_axes")),
        max_values_per_axis=_optional_int(payload.get("max_values_per_axis")),
        max_materialized_runs=_optional_int(payload.get("max_materialized_runs")),
    )


def submitted_run_from_wire(payload: Mapping[str, Any]) -> SubmittedRun:
    return SubmittedRun(
        id=str(payload["id"]) if payload.get("id") is not None else None,
        preset=str(payload["preset"]),
        dataset=str(payload["dataset"]),
        overrides=dict(_mapping(payload.get("overrides") or {})),
    )


def run_plan_from_wire(payload: Mapping[str, Any]) -> RunPlan:
    return RunPlan(
        identity=identity_from_wire(_mapping(payload["identity"])),
        presets=tuple(str(item) for item in payload["presets"]),
        experiment_task=str(payload["experiment_task"]),
        datasets=tuple(str(item) for item in payload["datasets"]),
        overrides=dict(_mapping(payload.get("overrides") or {})),
        search=search_spec_from_wire(payload.get("search")),
        runs=tuple(
            RunSpec(
                id=str(run["id"]),
                experiment_task=str(run["experiment_task"]),
                preset=str(run["preset"]),
                dataset=str(run["dataset"]),
                parameters=tuple(
                    RunParameter(
                        key=str(parameter["key"]),
                        value=parameter.get("value"),
                        source=parameter["source"],
                    )
                    for parameter in _mappings(run["parameters"])
                ),
            )
            for run in _mappings(payload["runs"])
        ),
    )


def package_metadata_to_wire(package: ModelPackage) -> dict[str, Any]:
    runtime_defaults = {
        key: to_wire(value)
        for key, value in vars(package.runtime_defaults).items()
        if key.isupper()
        and (value is None or isinstance(value, (str, int, float, bool, Enum, type)))
    }
    return {
        "identity": to_wire(package.identity),
        "catalog_key": package.catalog_key,
        "presets": [
            {
                "name": package.preset_name(preset),
                "key": preset.name,
                "label": preset.name,
                "description": package.preset_description(preset),
            }
            for preset in package.preset_type
        ],
        "default_experiment_task": package.task_name(package.default_experiment_task),
        "dataset_groups": [
            {
                "experiment_task": package.task_name(task),
                "label": package.task_label(task),
                "datasets": [
                    {
                        "name": dataset_name(dataset),
                        "label": dataset_label(dataset),
                        "input_dim": int(
                            getattr(dataset, "flattened_input_dim", 0) or 0
                        ),
                        "output_dim": int(getattr(dataset, "num_classes", 0) or 0),
                    }
                    for dataset in datasets
                ],
            }
            for task, datasets in package.dataset_metadata.items()
        ],
        "monitors": [option.to_api() for option in package.monitor_options()],
        "runtime_defaults": runtime_defaults,
    }


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Adapter value must be an object.")
    return value


def _mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Adapter value must be a list.")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError("Adapter list entries must be objects.")
    return list(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Adapter limit must be an integer.")
    return value


__all__ = [
    "PROTOCOL_VERSION",
    "configuration_schema_from_wire",
    "inspection_result_from_wire",
    "package_metadata_to_wire",
    "planning_budget_from_wire",
    "run_plan_from_wire",
    "run_request_from_wire",
    "search_space_from_wire",
    "submitted_run_from_wire",
    "to_wire",
]
