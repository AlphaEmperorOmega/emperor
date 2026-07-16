"""Public Interface for absolute positional embeddings."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.embedding.absolute._base import AbsolutePositionalEmbeddingBase
    from emperor.embedding.absolute._config import (
        AbsolutePositionalEmbeddingConfig,
        ImageLearnedPositionalEmbeddingConfig,
        ImageSinusoidalPositionalEmbeddingConfig,
        TextLearnedPositionalEmbeddingConfig,
        TextSinusoidalPositionalEmbeddingConfig,
    )
    from emperor.embedding.absolute._learned import (
        ImageLearnedPositionalEmbedding,
        LearnedPositionalEmbedding,
        TextLearnedPositionalEmbedding,
    )
    from emperor.embedding.absolute._sinusoidal import (
        ImageSinusoidalPositionalEmbedding,
        SinusoidalPositionalEmbedding,
        TextSinusoidalPositionalEmbedding,
    )
    from emperor.embedding.absolute._validation import (
        AbsolutePositionalEmbeddingValidator,
    )

__all__ = (
    "AbsolutePositionalEmbeddingConfig",
    "TextLearnedPositionalEmbeddingConfig",
    "ImageLearnedPositionalEmbeddingConfig",
    "TextSinusoidalPositionalEmbeddingConfig",
    "ImageSinusoidalPositionalEmbeddingConfig",
    "AbsolutePositionalEmbeddingBase",
    "LearnedPositionalEmbedding",
    "TextLearnedPositionalEmbedding",
    "ImageLearnedPositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "TextSinusoidalPositionalEmbedding",
    "ImageSinusoidalPositionalEmbedding",
    "AbsolutePositionalEmbeddingValidator",
)

_LAZY_EXPORTS = {
    "AbsolutePositionalEmbeddingConfig": (
        "emperor.embedding.absolute._config",
        "AbsolutePositionalEmbeddingConfig",
    ),
    "TextLearnedPositionalEmbeddingConfig": (
        "emperor.embedding.absolute._config",
        "TextLearnedPositionalEmbeddingConfig",
    ),
    "ImageLearnedPositionalEmbeddingConfig": (
        "emperor.embedding.absolute._config",
        "ImageLearnedPositionalEmbeddingConfig",
    ),
    "TextSinusoidalPositionalEmbeddingConfig": (
        "emperor.embedding.absolute._config",
        "TextSinusoidalPositionalEmbeddingConfig",
    ),
    "ImageSinusoidalPositionalEmbeddingConfig": (
        "emperor.embedding.absolute._config",
        "ImageSinusoidalPositionalEmbeddingConfig",
    ),
    "AbsolutePositionalEmbeddingBase": (
        "emperor.embedding.absolute._base",
        "AbsolutePositionalEmbeddingBase",
    ),
    "LearnedPositionalEmbedding": (
        "emperor.embedding.absolute._learned",
        "LearnedPositionalEmbedding",
    ),
    "TextLearnedPositionalEmbedding": (
        "emperor.embedding.absolute._learned",
        "TextLearnedPositionalEmbedding",
    ),
    "ImageLearnedPositionalEmbedding": (
        "emperor.embedding.absolute._learned",
        "ImageLearnedPositionalEmbedding",
    ),
    "SinusoidalPositionalEmbedding": (
        "emperor.embedding.absolute._sinusoidal",
        "SinusoidalPositionalEmbedding",
    ),
    "TextSinusoidalPositionalEmbedding": (
        "emperor.embedding.absolute._sinusoidal",
        "TextSinusoidalPositionalEmbedding",
    ),
    "ImageSinusoidalPositionalEmbedding": (
        "emperor.embedding.absolute._sinusoidal",
        "ImageSinusoidalPositionalEmbedding",
    ),
    "AbsolutePositionalEmbeddingValidator": (
        "emperor.embedding.absolute._validation",
        "AbsolutePositionalEmbeddingValidator",
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
