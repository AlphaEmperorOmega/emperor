"""One-way compatibility Adapters for selected Model Package metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from emperor.model_packages import (
    ModelMetadata,
    ModelPackage,
    discover_model_ids,
    is_safe_model_id,
    model_package,
)

from workbench.backend.inspection_serialization import (
    model_datasets_payload,
    model_monitors_payload,
    model_presets_payload,
)
from workbench.backend.inspector.errors import InspectorError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass(frozen=True)
class ModelParts:
    """Legacy projection of a selected Model Package."""

    name: str
    package: ModelPackage
    metadata: ModelMetadata
    config_module: ModuleType
    dataset_options_module: ModuleType
    monitor_options_module: ModuleType
    search_space_module: ModuleType
    presets_module: ModuleType
    model_module: ModuleType
    experiment_preset_enum: type[Enum]
    presets: Any
    experiment_type: type
    model_type: type[Any]
    default_experiment_task: Any
    dataset_options_by_task: dict[Any, list[type]]


def discover_models() -> list[str]:
    return discover_model_ids()


def load_model_parts(model_name: str) -> ModelParts:
    if not is_safe_model_id(model_name):
        raise InspectorError(f"Invalid model name: {model_name!r}")
    package = model_package(model_name)
    if package is None:
        raise InspectorError(f"Unknown model: {model_name}")
    try:
        metadata = package.metadata
        datasets = metadata.dataset_options_by_task
        parts = ModelParts(
            name=model_name,
            package=package,
            metadata=metadata,
            config_module=package.runtime_defaults,
            dataset_options_module=metadata.dataset_options_module,
            monitor_options_module=metadata.monitor_options_module,
            search_space_module=metadata.search_space_module,
            presets_module=package.presets_module,
            model_module=package.model_module,
            experiment_preset_enum=package.preset_type,
            presets=package.presets,
            experiment_type=package.experiment_type,
            model_type=package.model_class,
            default_experiment_task=metadata.default_experiment_task,
            dataset_options_by_task=datasets,
        )
    except Exception as exc:
        raise InspectorError(
            f"Failed to import model package '{model_name}': {exc}"
        ) from exc
    if not datasets:
        raise InspectorError(
            f"Model package '{model_name}' does not define DATASET_OPTIONS_BY_TASK."
        )
    return parts


def list_model_monitors(model_name: str) -> list[dict[str, Any]]:
    return model_monitors_payload(load_model_parts(model_name).package)


def list_model_datasets(model_name: str) -> dict[str, Any]:
    return model_datasets_payload(load_model_parts(model_name).package)


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    return model_presets_payload(load_model_parts(model_name).package)


__all__ = [
    "ModelParts",
    "discover_models",
    "list_model_datasets",
    "list_model_monitors",
    "list_model_presets",
    "load_model_parts",
]
