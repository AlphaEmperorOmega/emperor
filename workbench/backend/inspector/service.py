"""One-way compatibility Adapter for the Emperor Inspection Interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from emperor.inspection import InspectionRequest, ParsedOverrides
from emperor.inspection import inspect_model as inspect_model_semantically

from workbench.backend.inspection_errors import call_inspection
from workbench.backend.inspection_serialization import inspection_result_payload
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
    package = load_model_parts(model_name).package
    semantic = call_inspection(
        inspect_model_semantically,
        package,
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
    return inspection_result_payload(semantic)


__all__ = ["inspect_model"]
