"""Public Interface for relative positional embeddings."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.embedding.relative._bias import DynamicPositionalBias
    from emperor.embedding.relative._config import (
        DynamicPositionalBiasConfig,
        RelativePositionalEmbeddingConfig,
    )
    from emperor.embedding.relative._validation import (
        RelativePositionalEmbeddingValidator,
    )

__all__ = (
    "RelativePositionalEmbeddingConfig",
    "DynamicPositionalBiasConfig",
    "DynamicPositionalBias",
    "RelativePositionalEmbeddingValidator",
)

_LAZY_EXPORTS = {
    "RelativePositionalEmbeddingConfig": (
        "emperor.embedding.relative._config",
        "RelativePositionalEmbeddingConfig",
    ),
    "DynamicPositionalBiasConfig": (
        "emperor.embedding.relative._config",
        "DynamicPositionalBiasConfig",
    ),
    "DynamicPositionalBias": (
        "emperor.embedding.relative._bias",
        "DynamicPositionalBias",
    ),
    "RelativePositionalEmbeddingValidator": (
        "emperor.embedding.relative._validation",
        "RelativePositionalEmbeddingValidator",
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
