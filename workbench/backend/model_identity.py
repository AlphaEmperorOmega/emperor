from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.packages import model_key, split_model_id
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.project_adapter import ProjectAdapterFailure, project_adapter


def require_model_id(model_type: str, model: str) -> str:
    try:
        model_id = model_key(model_type, model)
        project_adapter().package(model_id)
    except (ProjectAdapterFailure, ValueError) as exc:
        raise InspectionFailure(
            f"Unknown model: --model-type {model_type} --model {model}"
        ) from exc
    return model_id


def model_identity_payload_from_id(model_id: str) -> dict[str, str]:
    identity = split_model_id(model_id)
    if identity is None:
        raise ValueError(f"Invalid model id: {model_id}")
    return identity.to_payload()


def model_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    model_type = payload.get("modelType")
    model = payload.get("model")
    if isinstance(model_type, str) and isinstance(model, str):
        try:
            return model_key(model_type, model)
        except ValueError:
            return None
    if not isinstance(model, str):
        return None
    identity = split_model_id(model)
    if identity is not None:
        return identity.catalog_key
    try:
        candidates = [
            package.catalog_key
            for package in project_adapter().catalog()
            if package.model == model
        ]
    except ProjectAdapterFailure:
        return None
    return min(candidates, key=_flat_name_priority) if candidates else None


def _flat_name_priority(model_id: str) -> tuple[int, str]:
    if model_id.startswith("linears/"):
        return (0, model_id)
    if model_id.startswith("experts/"):
        return (1, model_id)
    if model_id.startswith("bert/"):
        return (2, model_id)
    if model_id.startswith("vit/"):
        return (3, model_id)
    if model_id.startswith("neuron/"):
        return (4, model_id)
    return (5, model_id)


def normalize_preset_token(preset: str | None) -> str | None:
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


__all__ = [
    "model_id_from_payload",
    "model_identity_payload_from_id",
    "normalize_preset_token",
    "require_model_id",
]
