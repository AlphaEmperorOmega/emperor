from __future__ import annotations

from types import ModuleType

from model_runtime.packages.metadata import ModelMetadata
from model_runtime.packages.metadata import (
    load_model_metadata_for_config_module as _load_for_config_module,
)
from model_runtime.packages.metadata import (
    load_model_metadata_from_module_path as _load_from_module_path,
)


def load_model_metadata_from_module_path(
    module_path: str,
    *,
    model_name: str | None = None,
) -> ModelMetadata:
    return _load_from_module_path(module_path, model_name=model_name)


def load_model_metadata(model_name: str) -> ModelMetadata:
    from models.catalog import model_package

    package = model_package(model_name)
    if package is None:
        raise ValueError(f"Unknown model: {model_name}")
    return package.metadata


def load_model_metadata_for_config_module(config_module: ModuleType) -> ModelMetadata:
    return _load_for_config_module(config_module)


__all__ = [
    "ModelMetadata",
    "load_model_metadata",
    "load_model_metadata_for_config_module",
    "load_model_metadata_from_module_path",
]
