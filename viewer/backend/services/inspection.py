"""Model inspection use cases."""

from __future__ import annotations

from typing import Any

from models.catalog import model_id_from_parts

from viewer.backend.inspector.errors import InspectorError


def _model_id(model_type: str, model: str) -> str:
    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        raise InspectorError(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return model_id


class InspectionService:
    def inspect(
        self,
        *,
        model_type: str,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
    ) -> dict[str, Any]:
        from viewer.backend.inspector.service import inspect_model

        return inspect_model(
            _model_id(model_type, model),
            preset,
            overrides,
            dataset=dataset,
        )
