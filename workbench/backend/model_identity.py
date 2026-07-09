"""Shared model identity helpers for Workbench request handling."""

from __future__ import annotations

from models.catalog import model_id_from_parts

from workbench.backend.inspector.errors import InspectorError


def require_model_id(model_type: str, model: str) -> str:
    """Return the internal model id or raise the stable inspector error."""

    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        raise InspectorError(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return model_id


__all__ = ["require_model_id"]
