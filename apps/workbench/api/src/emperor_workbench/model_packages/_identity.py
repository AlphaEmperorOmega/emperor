from __future__ import annotations

from collections.abc import Iterable

from emperor_workbench.model_packages._records import ModelPackageIdentity


def normalize_preset_token(preset: str | None) -> str | None:
    if preset is None:
        return None
    return str(preset).lower().replace("_", "-")


def flat_model_identity(
    model: str,
    identities: Iterable[ModelPackageIdentity],
) -> ModelPackageIdentity | None:
    candidates = [identity for identity in identities if identity.model == model]
    return min(candidates, key=_flat_name_priority) if candidates else None


def _flat_name_priority(identity: ModelPackageIdentity) -> tuple[int, str]:
    model_id = identity.catalog_key
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


__all__ = ["normalize_preset_token"]
