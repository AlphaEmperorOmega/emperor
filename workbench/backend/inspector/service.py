"""One-way compatibility Adapter for the Emperor Inspection Interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from emperor.inspection import InspectionRequest, ParsedOverrides

from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspector.discovery import load_model_parts


def inspect_model(
    model_name: str,
    preset_name: str,
    overrides: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    experiment_task: str | None = None,
    *,
    parsed_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = WorkbenchInspectionAdapter.from_package(
        load_model_parts(model_name).package
    )
    return adapter.inspect_payload(
        InspectionRequest(
            preset=preset_name,
            overrides=(
                ParsedOverrides(parsed_overrides)
                if parsed_overrides is not None
                else (overrides or {})
            ),
            dataset=dataset,
            experiment_task=experiment_task,
        ),
    )


__all__ = ["inspect_model"]
