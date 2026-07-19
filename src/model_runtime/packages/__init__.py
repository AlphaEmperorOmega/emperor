from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model_runtime.packages.configuration import (
        ConfigValueError,
        abstract_config_class_error,
        canonical_config_key,
        canonical_config_key_for_module,
        config_key_to_flag,
        config_key_to_model_param,
        config_key_to_param,
        iter_supported_config_keys,
        normalize_key,
        parse_config_value,
        search_key_to_config_key,
        serialize_config_value,
    )
    from model_runtime.packages.configuration_metadata import (
        configuration_field_metadata,
    )
    from model_runtime.packages.datasets import (
        dataset_class_name_to_cli_name,
        dataset_cli_name,
        dataset_label,
        dataset_name,
        normalize_dataset_name,
    )
    from model_runtime.packages.definition import (
        ModelCatalogEntry,
        ModelPackage,
        model_package_from_module_path,
    )
    from model_runtime.packages.identity import (
        MODEL_ID_SEGMENT_RE,
        ModelIdentity,
        is_safe_model_identity,
        is_safe_model_segment,
        model_key,
        split_model_id,
    )
    from model_runtime.packages.inspection_limits import (
        DEFAULT_INSPECTION_CONSTRUCTION_LIMITS,
        InspectionConstructionLimits,
    )
    from model_runtime.packages.metadata import (
        ModelMetadata,
        load_model_metadata_for_config_module,
        load_model_metadata_from_module_path,
    )
    from model_runtime.packages.presets import (
        BuilderBackedExperimentPresetsBase,
        ExperimentPresetsBase,
        GridSearch,
        PresetDefinition,
        PresetLock,
        RandomSearch,
        SearchMode,
        create_search_space,
    )

__all__ = [
    "DEFAULT_INSPECTION_CONSTRUCTION_LIMITS",
    "MODEL_ID_SEGMENT_RE",
    "BuilderBackedExperimentPresetsBase",
    "ConfigValueError",
    "ExperimentPresetsBase",
    "GridSearch",
    "InspectionConstructionLimits",
    "ModelCatalogEntry",
    "ModelIdentity",
    "ModelMetadata",
    "ModelPackage",
    "PresetDefinition",
    "PresetLock",
    "RandomSearch",
    "SearchMode",
    "abstract_config_class_error",
    "canonical_config_key",
    "canonical_config_key_for_module",
    "config_key_to_flag",
    "config_key_to_model_param",
    "config_key_to_param",
    "configuration_field_metadata",
    "create_search_space",
    "dataset_class_name_to_cli_name",
    "dataset_cli_name",
    "dataset_label",
    "dataset_name",
    "is_safe_model_identity",
    "is_safe_model_segment",
    "iter_supported_config_keys",
    "load_model_metadata_for_config_module",
    "load_model_metadata_from_module_path",
    "model_key",
    "model_package_from_module_path",
    "normalize_dataset_name",
    "normalize_key",
    "parse_config_value",
    "search_key_to_config_key",
    "serialize_config_value",
    "split_model_id",
]

_CONFIGURATION_EXPORTS = {
    "ConfigValueError",
    "abstract_config_class_error",
    "canonical_config_key",
    "canonical_config_key_for_module",
    "config_key_to_flag",
    "config_key_to_model_param",
    "config_key_to_param",
    "iter_supported_config_keys",
    "normalize_key",
    "parse_config_value",
    "search_key_to_config_key",
    "serialize_config_value",
}
_CONFIGURATION_METADATA_EXPORTS = {"configuration_field_metadata"}
_DATASET_EXPORTS = {
    "dataset_class_name_to_cli_name",
    "dataset_cli_name",
    "dataset_label",
    "dataset_name",
    "normalize_dataset_name",
}
_DEFINITION_EXPORTS = {
    "ModelCatalogEntry",
    "ModelPackage",
    "model_package_from_module_path",
}
_IDENTITY_EXPORTS = {
    "MODEL_ID_SEGMENT_RE",
    "ModelIdentity",
    "is_safe_model_identity",
    "is_safe_model_segment",
    "model_key",
    "split_model_id",
}
_INSPECTION_LIMIT_EXPORTS = {
    "DEFAULT_INSPECTION_CONSTRUCTION_LIMITS",
    "InspectionConstructionLimits",
}
_METADATA_EXPORTS = {
    "ModelMetadata",
    "load_model_metadata_for_config_module",
    "load_model_metadata_from_module_path",
}
_PRESET_EXPORTS = {
    "BuilderBackedExperimentPresetsBase",
    "ExperimentPresetsBase",
    "GridSearch",
    "PresetDefinition",
    "PresetLock",
    "RandomSearch",
    "SearchMode",
    "create_search_space",
}


def __getattr__(name: str) -> Any:
    if name in _CONFIGURATION_EXPORTS:
        module_name = "model_runtime.packages.configuration"
    elif name in _CONFIGURATION_METADATA_EXPORTS:
        module_name = "model_runtime.packages.configuration_metadata"
    elif name in _DATASET_EXPORTS:
        module_name = "model_runtime.packages.datasets"
    elif name in _DEFINITION_EXPORTS:
        module_name = "model_runtime.packages.definition"
    elif name in _IDENTITY_EXPORTS:
        module_name = "model_runtime.packages.identity"
    elif name in _INSPECTION_LIMIT_EXPORTS:
        module_name = "model_runtime.packages.inspection_limits"
    elif name in _METADATA_EXPORTS:
        module_name = "model_runtime.packages.metadata"
    elif name in _PRESET_EXPORTS:
        module_name = "model_runtime.packages.presets"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
