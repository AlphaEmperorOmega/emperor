from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import InspectionRequest, ParsedOverrides
from model_runtime.packages import is_safe_model_identity
from workbench.backend.inspection_adapter import (
    WorkbenchInspectionAdapter,
    inspect_module_graph_payload,
)
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.project_adapter import project_adapter


def discover_models() -> list[str]:
    return [package.catalog_key for package in project_adapter().catalog()]


def _adapter(model_name: str) -> WorkbenchInspectionAdapter:
    parts = model_name.split("/")
    if len(parts) != 2 or not is_safe_model_identity(parts[0], parts[1]):
        raise InspectionFailure(f"Invalid model name: {model_name!r}")
    return WorkbenchInspectionAdapter.select(model_name)


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    return _adapter(model_name).configuration_payload(preset_name)


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    return _adapter(model_name).search_space_payload(preset_name, preset_names)


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    return _adapter(model_name).presets_payload()


def list_model_datasets(model_name: str) -> dict[str, Any]:
    return _adapter(model_name).datasets_payload()


def list_model_monitors(model_name: str) -> list[dict[str, Any]]:
    return _adapter(model_name).monitors_payload()


def inspect_model(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    experiment_task: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _adapter(model_name).inspect_payload(
        InspectionRequest(
            preset=preset_name,
            overrides=(
                ParsedOverrides(parsed_overrides)
                if parsed_overrides is not None
                else (overrides or {})
            ),
            dataset=dataset,
            experiment_task=experiment_task,
        )
    )


serialize_graph = inspect_module_graph_payload


__all__ = [
    "config_schema",
    "discover_models",
    "inspect_model",
    "list_model_datasets",
    "list_model_monitors",
    "list_model_presets",
    "search_space_schema",
    "serialize_graph",
]
