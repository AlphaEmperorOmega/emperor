"""Shared model identity helpers for Workbench request handling."""

from __future__ import annotations

from emperor.model_packages import model_id_from_parts

from workbench.backend.inspection_errors import InspectionFailure


def require_model_id(model_type: str, model: str) -> str:
    """Return the internal model id or raise the stable inspector error."""

    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        raise InspectionFailure(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return model_id


def normalize_preset_token(preset: str | None) -> str | None:
    """Normalize the equivalent underscore/hyphen preset spellings."""
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


__all__ = [
    "normalize_preset_token",
    "require_model_id",
]
