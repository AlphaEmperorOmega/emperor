from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from inspect import cleandoc
from typing import Any, cast

from emperor.config import ConfigBase
from model_runtime.inspection._graph_accounting import GraphModule
from model_runtime.inspection._graph_semantic_catalog import (
    DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
    DEFAULT_RESIDUAL_OPTION_DESCRIPTION,
    PROJECT_MODEL_DESCRIPTION,
    SEMANTIC_TYPE_CATALOG,
)
from model_runtime.inspection.capture_limits import InspectionCaptureLimits
from model_runtime.inspection.errors import InspectionError
from model_runtime.inspection.records import (
    GraphConfiguration,
    GraphConfigurationField,
    GraphRole,
)

ARCHITECTURE_ROLE: GraphRole = "architecture"
INTERNAL_ROLE: GraphRole = "internal"
RUNTIME_ROLE: GraphRole = "runtime"


class _MissingAttribute:
    __slots__ = ()


_MISSING_ATTRIBUTE = _MissingAttribute()


def display_graph_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    return value


def _config_field_value(value: Any) -> Any:
    if value is None:
        rendered: Any = None
    elif isinstance(value, Enum):
        rendered = value.name
    elif isinstance(value, ConfigBase):
        rendered = type(value).__name__
    elif isinstance(value, type):
        rendered = value.__name__
    elif isinstance(value, (str, int, float, bool)):
        rendered = value
    elif is_dataclass(value):
        rendered = type(value).__name__
    else:
        rendered = str(value)
    return rendered


def _module_config_instance(module: GraphModule) -> Any | None:
    config = getattr(module, "_emperor_config", None)
    if config is None:
        config = getattr(module, "cfg", None)
    if config is None or isinstance(config, type) or not is_dataclass(config):
        return None
    return config


def _metadata_help(metadata: Any) -> str | None:
    metadata_get = getattr(metadata, "get", _MISSING_ATTRIBUTE)
    help_text = (
        cast(Any, metadata_get)("help")
        if metadata_get is not _MISSING_ATTRIBUTE
        else None
    )
    if not isinstance(help_text, str):
        return None
    help_text = help_text.strip()
    return help_text or None


def _flattened_residual_configuration_fields(
    config: Any,
) -> tuple[GraphConfigurationField, GraphConfigurationField | None]:
    residual_config: Any = config.residual_config
    residual_config_type: type[object] | None = (
        None if residual_config is None else type(cast(object, residual_config))
    )
    config_policy = SEMANTIC_TYPE_CATALOG.policy_for(type(cast(object, config)))
    descriptions = (
        config_policy.residual_field_descriptions if config_policy is not None else None
    )
    option_description, model_description = descriptions or (
        DEFAULT_RESIDUAL_OPTION_DESCRIPTION,
        DEFAULT_RESIDUAL_MODEL_DESCRIPTION,
    )
    residual_model_config = getattr(
        residual_config,
        "model_config",
        _MISSING_ATTRIBUTE,
    )
    has_residual_model_config = residual_model_config is not _MISSING_ATTRIBUTE
    return (
        GraphConfigurationField(
            key="residual_connection_option",
            value=_config_field_value(residual_config_type),
            description=option_description,
        ),
        (
            GraphConfigurationField(
                key="residual_model_config",
                value=_config_field_value(residual_model_config),
                description=model_description,
            )
            if has_residual_model_config
            else None
        ),
    )


def module_configuration(
    module: GraphModule,
    limits: InspectionCaptureLimits,
) -> GraphConfiguration | None:
    config = _module_config_instance(module)
    if config is None:
        return None

    config_fields = fields(config)
    if len(config_fields) > limits.maximum_configuration_fields:
        raise InspectionError(
            "Inspection configuration field limit of "
            f"{limits.maximum_configuration_fields} exceeded."
        )
    serialized_fields: list[GraphConfigurationField] = []
    flattened_residual_model_field: GraphConfigurationField | None = None

    def append_field(configuration_field: GraphConfigurationField) -> None:
        if len(serialized_fields) >= limits.maximum_configuration_fields:
            raise InspectionError(
                "Inspection configuration field limit of "
                f"{limits.maximum_configuration_fields} exceeded."
            )
        serialized_fields.append(configuration_field)

    for field in config_fields:
        if field.name == "residual_config":
            (
                residual_option_field,
                flattened_residual_model_field,
            ) = _flattened_residual_configuration_fields(config)
            append_field(residual_option_field)
            continue
        description = _metadata_help(field.metadata)
        append_field(
            GraphConfigurationField(
                key=field.name,
                value=_config_field_value(getattr(config, field.name)),
                description=description,
            )
        )
    if flattened_residual_model_field is not None:
        append_field(flattened_residual_model_field)

    return GraphConfiguration(
        type_name=type(config).__name__,
        fields=tuple(serialized_fields),
    )


def _explicit_docstring_description(class_type: type[Any]) -> str | None:
    raw_docstring = class_type.__dict__.get("__doc__")
    if not isinstance(raw_docstring, str):
        return None
    docstring = cleandoc(raw_docstring).strip()
    if not docstring or docstring.startswith(f"{class_type.__name__}("):
        return None
    return docstring.split("\n\n", 1)[0].replace("\n", " ")


def _project_model_description(module_type: type[Any]) -> str | None:
    module_name = module_type.__module__
    if (
        SEMANTIC_TYPE_CATALOG.is_registered_type(module_type)
        and module_type.__name__ == "Model"
        and module_type.__qualname__ == "Model"
        and module_name.startswith("models.")
        and module_name.endswith(".model")
    ):
        return PROJECT_MODEL_DESCRIPTION
    return None


def _catalog_description(class_type: type[object]) -> str | None:
    policy = SEMANTIC_TYPE_CATALOG.policy_for(class_type)
    return policy.description if policy is not None else None


def component_description(module: GraphModule) -> str | None:
    module_type: type[object] = type(module)
    description = _catalog_description(module_type)
    if description is None:
        description = _project_model_description(module_type)
    if description is not None:
        return description

    config = _module_config_instance(module)
    if config is not None:
        config_type = type(cast(object, config))
        description = _catalog_description(config_type)
        if description is not None:
            return description

    if not module_type.__module__.startswith("torch."):
        description = _explicit_docstring_description(module_type)
        if description is not None:
            return description
    if config is not None:
        return _explicit_docstring_description(type(cast(object, config)))
    return None


def component_graph_role(module: GraphModule) -> GraphRole:
    module_type = type(module)
    policy = SEMANTIC_TYPE_CATALOG.policy_for(module_type)
    if policy is not None:
        return policy.graph_role
    if module_type.__module__.startswith("torchmetrics."):
        return RUNTIME_ROLE
    return ARCHITECTURE_ROLE


__all__ = [
    "ARCHITECTURE_ROLE",
    "INTERNAL_ROLE",
    "RUNTIME_ROLE",
    "component_description",
    "component_graph_role",
    "display_graph_value",
    "module_configuration",
]
