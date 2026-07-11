"""Shared model identity helpers for Workbench request handling."""

from __future__ import annotations

from emperor.model_packages import ModelPackage, model_id_from_parts, model_package

from workbench.backend.inspector.errors import InspectorError


def require_model_id(model_type: str, model: str) -> str:
    """Return the internal model id or raise the stable inspector error."""

    model_id = model_id_from_parts(model_type, model)
    if model_id is None:
        raise InspectorError(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return model_id


def require_model_package(model_type: str, model: str) -> ModelPackage:
    model_id = require_model_id(model_type, model)
    package = model_package(model_id)
    if package is None:  # The identity and catalog are one atomic Interface.
        raise InspectorError(
            f"Unknown model: --model-type {model_type} --model {model}"
        )
    return package


def normalize_preset_token(preset: str | None) -> str | None:
    """Normalize the equivalent underscore/hyphen preset spellings."""
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


__all__ = [
    "normalize_preset_token",
    "require_model_id",
    "require_model_package",
]
