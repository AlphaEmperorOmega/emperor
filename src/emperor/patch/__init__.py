"""Public Interface for image patch embeddings."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.patch._base import PatchBase
    from emperor.patch._config import (
        ConvPatchEmbeddingConfig,
        LinearPatchEmbeddingConfig,
        PatchConfig,
    )
    from emperor.patch._variants.convolutional import PatchEmbeddingConv
    from emperor.patch._variants.linear import PatchEmbeddingLinear

__all__ = (
    "PatchConfig",
    "LinearPatchEmbeddingConfig",
    "ConvPatchEmbeddingConfig",
    "PatchBase",
    "PatchEmbeddingLinear",
    "PatchEmbeddingConv",
)

_LAZY_EXPORTS = {
    "PatchConfig": ("emperor.patch._config", "PatchConfig"),
    "LinearPatchEmbeddingConfig": (
        "emperor.patch._config",
        "LinearPatchEmbeddingConfig",
    ),
    "ConvPatchEmbeddingConfig": (
        "emperor.patch._config",
        "ConvPatchEmbeddingConfig",
    ),
    "PatchBase": ("emperor.patch._base", "PatchBase"),
    "PatchEmbeddingLinear": (
        "emperor.patch._variants.linear",
        "PatchEmbeddingLinear",
    ),
    "PatchEmbeddingConv": (
        "emperor.patch._variants.convolutional",
        "PatchEmbeddingConv",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
