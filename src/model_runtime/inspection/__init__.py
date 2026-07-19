from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_runtime.inspection.errors import InspectionError
    from model_runtime.inspection.field_descriptions import config_field_description
    from model_runtime.inspection.model_graph import (
        ARCHITECTURE_ROLE,
        INTERNAL_ROLE,
        ROOT_NODE_ID,
        ROOT_NODE_PATH,
        RUNTIME_ROLE,
        graph_role,
        inspect_model_graph,
        module_details,
        parameter_count,
        parameter_size_bytes,
    )
    from model_runtime.inspection.overrides import (
        canonicalize_overrides,
        parse_overrides,
        reject_locked_overrides,
        resolve_override_key,
        serialize_overrides,
        supported_config_keys,
    )
    from model_runtime.inspection.records import (
        ConfigurationField,
        ConfigurationSchema,
        GraphConfiguration,
        GraphConfigurationField,
        GraphEdge,
        GraphNode,
        GraphRole,
        InspectionRequest,
        InspectionResult,
        ModelGraph,
        ParsedOverrides,
        SearchAxis,
        SearchSpace,
    )
    from model_runtime.inspection.schema import (
        configuration_schema,
        preset_locks,
        search_space_schema,
    )
    from model_runtime.inspection.service import inspect_model, validate_configuration
    from model_runtime.inspection.shape_trace import (
        MethodShapeTrace,
        ModelShapeTrace,
        ModuleShapeCall,
        ModuleShapeTrace,
        ShapeTraceDetail,
        TensorShape,
        TensorVariableTrace,
        inspect_model_shapes,
    )

__all__ = [
    "ARCHITECTURE_ROLE",
    "ConfigurationField",
    "ConfigurationSchema",
    "GraphConfiguration",
    "GraphConfigurationField",
    "GraphEdge",
    "GraphNode",
    "GraphRole",
    "INTERNAL_ROLE",
    "InspectionError",
    "InspectionRequest",
    "InspectionResult",
    "MethodShapeTrace",
    "ModelGraph",
    "ModelShapeTrace",
    "ModuleShapeCall",
    "ModuleShapeTrace",
    "ParsedOverrides",
    "ROOT_NODE_ID",
    "ROOT_NODE_PATH",
    "RUNTIME_ROLE",
    "SearchAxis",
    "SearchSpace",
    "ShapeTraceDetail",
    "TensorShape",
    "TensorVariableTrace",
    "canonicalize_overrides",
    "configuration_schema",
    "config_field_description",
    "graph_role",
    "inspect_model_graph",
    "inspect_model_shapes",
    "inspect_model",
    "module_details",
    "parameter_count",
    "parameter_size_bytes",
    "parse_overrides",
    "preset_locks",
    "reject_locked_overrides",
    "resolve_override_key",
    "search_space_schema",
    "serialize_overrides",
    "supported_config_keys",
    "validate_configuration",
]


def __getattr__(name: str) -> Any:
    if name == "InspectionError":
        from model_runtime.inspection.errors import InspectionError

        return InspectionError
    if name in {
        "ConfigurationField",
        "ConfigurationSchema",
        "GraphConfiguration",
        "GraphConfigurationField",
        "GraphEdge",
        "GraphNode",
        "GraphRole",
        "InspectionRequest",
        "InspectionResult",
        "ModelGraph",
        "ParsedOverrides",
        "SearchAxis",
        "SearchSpace",
    }:
        from model_runtime.inspection import records

        return getattr(records, name)
    if name in {
        "MethodShapeTrace",
        "ModelShapeTrace",
        "ModuleShapeCall",
        "ModuleShapeTrace",
        "ShapeTraceDetail",
        "TensorShape",
        "TensorVariableTrace",
        "inspect_model_shapes",
    }:
        from model_runtime.inspection import shape_trace

        return getattr(shape_trace, name)
    if name in {
        "ARCHITECTURE_ROLE",
        "INTERNAL_ROLE",
        "ROOT_NODE_ID",
        "ROOT_NODE_PATH",
        "RUNTIME_ROLE",
        "graph_role",
        "inspect_model_graph",
        "module_details",
        "parameter_count",
        "parameter_size_bytes",
    }:
        from model_runtime.inspection import model_graph

        return getattr(model_graph, name)
    if name == "inspect_model":
        from model_runtime.inspection.service import inspect_model

        return inspect_model
    if name == "validate_configuration":
        from model_runtime.inspection.service import validate_configuration

        return validate_configuration
    if name == "configuration_schema":
        from model_runtime.inspection.schema import configuration_schema

        return configuration_schema
    if name == "config_field_description":
        from model_runtime.inspection.field_descriptions import config_field_description

        return config_field_description
    if name == "search_space_schema":
        from model_runtime.inspection.schema import search_space_schema

        return search_space_schema
    if name == "preset_locks":
        from model_runtime.inspection.schema import preset_locks

        return preset_locks
    if name == "parse_overrides":
        from model_runtime.inspection.overrides import parse_overrides

        return parse_overrides
    if name in {
        "canonicalize_overrides",
        "reject_locked_overrides",
        "resolve_override_key",
        "serialize_overrides",
        "supported_config_keys",
    }:
        from model_runtime.inspection import overrides

        return getattr(overrides, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
