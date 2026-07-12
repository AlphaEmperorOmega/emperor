"""Compatibility Adapters for Emperor configuration and search schemas."""

from __future__ import annotations

from typing import Any

from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter


def preset_locks(model_name: str, preset_name: str | None) -> dict[str, Any]:
    return WorkbenchInspectionAdapter.select(model_name).preset_locks(preset_name)


def config_schema(model_name: str, preset_name: str | None = None) -> dict[str, Any]:
    return WorkbenchInspectionAdapter.select(model_name).configuration_payload(
        preset_name
    )


def search_space_schema(
    model_name: str,
    preset_name: str | None = None,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    return WorkbenchInspectionAdapter.select(model_name).search_space_payload(
        preset_name,
        preset_names,
    )


__all__ = [
    "config_schema",
    "preset_locks",
    "search_space_schema",
]
