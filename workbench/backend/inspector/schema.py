"""Compatibility Adapters for Emperor configuration and search schemas."""

from __future__ import annotations

from typing import Any

from emperor.inspection import (
    configuration_schema,
)
from emperor.inspection import (
    preset_locks as semantic_preset_locks,
)
from emperor.inspection import (
    search_space_schema as semantic_search_space_schema,
)
from emperor.model_packages import model_package

from workbench.backend.inspection_errors import call_inspection
from workbench.backend.inspection_serialization import (
    configuration_schema_payload,
    search_space_payload,
)
from workbench.backend.inspector.errors import InspectorError


def _package(model_name: str):
    package = model_package(model_name)
    if package is None:
        raise InspectorError(f"Unknown model: {model_name}")
    return package


def preset_locks(model_name: str, preset_name: str | None) -> dict[str, Any]:
    return call_inspection(
        semantic_preset_locks,
        _package(model_name),
        preset_name,
    )


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    return configuration_schema_payload(
        call_inspection(
            configuration_schema,
            _package(model_name),
            preset_name,
        )
    )


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    return search_space_payload(
        call_inspection(
            semantic_search_space_schema,
            _package(model_name),
            preset_name,
            preset_names,
        )
    )


__all__ = [
    "config_schema",
    "preset_locks",
    "search_space_schema",
]
