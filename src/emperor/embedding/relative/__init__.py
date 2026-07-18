"""Public Interface for relative positional embedding configuration."""

from emperor.embedding.relative._config import (
    DynamicPositionalBiasConfig,
    RelativePositionalEmbeddingConfig,
)

__all__ = (
    "RelativePositionalEmbeddingConfig",
    "DynamicPositionalBiasConfig",
)
