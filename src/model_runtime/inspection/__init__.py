from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_runtime.inspection.capture_limits import InspectionCaptureLimits
    from model_runtime.inspection.errors import InspectionError
    from model_runtime.inspection.field_descriptions import config_field_description
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
        ConfigurationFieldCondition,
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

_EXPORT_MODULE_BY_NAME = {
    "ConfigurationField": "model_runtime.inspection.records",
    "ConfigurationFieldCondition": "model_runtime.inspection.records",
    "ConfigurationSchema": "model_runtime.inspection.records",
    "GraphConfiguration": "model_runtime.inspection.records",
    "GraphConfigurationField": "model_runtime.inspection.records",
    "GraphEdge": "model_runtime.inspection.records",
    "GraphNode": "model_runtime.inspection.records",
    "GraphRole": "model_runtime.inspection.records",
    "InspectionError": "model_runtime.inspection.errors",
    "InspectionCaptureLimits": "model_runtime.inspection.capture_limits",
    "InspectionRequest": "model_runtime.inspection.records",
    "InspectionResult": "model_runtime.inspection.records",
    "MethodShapeTrace": "model_runtime.inspection.shape_trace",
    "ModelGraph": "model_runtime.inspection.records",
    "ModelShapeTrace": "model_runtime.inspection.shape_trace",
    "ModuleShapeCall": "model_runtime.inspection.shape_trace",
    "ModuleShapeTrace": "model_runtime.inspection.shape_trace",
    "ParsedOverrides": "model_runtime.inspection.records",
    "SearchAxis": "model_runtime.inspection.records",
    "SearchSpace": "model_runtime.inspection.records",
    "ShapeTraceDetail": "model_runtime.inspection.shape_trace",
    "TensorShape": "model_runtime.inspection.shape_trace",
    "TensorVariableTrace": "model_runtime.inspection.shape_trace",
    "canonicalize_overrides": "model_runtime.inspection.overrides",
    "configuration_schema": "model_runtime.inspection.schema",
    "config_field_description": "model_runtime.inspection.field_descriptions",
    "inspect_model_shapes": "model_runtime.inspection.shape_trace",
    "inspect_model": "model_runtime.inspection.service",
    "parse_overrides": "model_runtime.inspection.overrides",
    "preset_locks": "model_runtime.inspection.schema",
    "reject_locked_overrides": "model_runtime.inspection.overrides",
    "resolve_override_key": "model_runtime.inspection.overrides",
    "search_space_schema": "model_runtime.inspection.schema",
    "serialize_overrides": "model_runtime.inspection.overrides",
    "supported_config_keys": "model_runtime.inspection.overrides",
    "validate_configuration": "model_runtime.inspection.service",
}

__all__ = [
    "ConfigurationField",
    "ConfigurationFieldCondition",
    "ConfigurationSchema",
    "GraphConfiguration",
    "GraphConfigurationField",
    "GraphEdge",
    "GraphNode",
    "GraphRole",
    "InspectionError",
    "InspectionCaptureLimits",
    "InspectionRequest",
    "InspectionResult",
    "MethodShapeTrace",
    "ModelGraph",
    "ModelShapeTrace",
    "ModuleShapeCall",
    "ModuleShapeTrace",
    "ParsedOverrides",
    "SearchAxis",
    "SearchSpace",
    "ShapeTraceDetail",
    "TensorShape",
    "TensorVariableTrace",
    "canonicalize_overrides",
    "configuration_schema",
    "config_field_description",
    "inspect_model_shapes",
    "inspect_model",
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
    module_name = _EXPORT_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
