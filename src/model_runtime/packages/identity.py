from __future__ import annotations

import re
from dataclasses import dataclass

MODEL_ID_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelIdentity:
    """Stable Model Package identity, independent from its Python module path."""

    model_type: str
    model: str

    def __post_init__(self) -> None:
        model_key(self.model_type, self.model)

    @property
    def catalog_key(self) -> str:
        return model_key(self.model_type, self.model)

    def to_payload(self) -> dict[str, str]:
        return {
            "modelType": self.model_type,
            "model": self.model,
        }


def is_safe_model_segment(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value or value.strip() != value:
        return False
    return bool(MODEL_ID_SEGMENT_RE.fullmatch(value))


def is_safe_model_identity(model_type: str, model: str) -> bool:
    return is_safe_model_segment(model_type) and is_safe_model_segment(model)


def model_key(model_type: str, model: str) -> str:
    if not is_safe_model_identity(model_type, model):
        raise ValueError(f"Invalid model identity: {model_type!r}/{model!r}")
    return f"{model_type}/{model}"


def split_model_id(model_id: object) -> ModelIdentity | None:
    if not isinstance(model_id, str) or "\\" in model_id:
        return None
    segments = model_id.split("/")
    if len(segments) != 2 or not all(is_safe_model_segment(item) for item in segments):
        return None
    return ModelIdentity(model_type=segments[0], model=segments[1])


__all__ = [
    "MODEL_ID_SEGMENT_RE",
    "ModelIdentity",
    "is_safe_model_identity",
    "is_safe_model_segment",
    "model_key",
    "split_model_id",
]
