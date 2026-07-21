"""Public Interface for Transformer modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.transformer._config import (
        TransformerConfig,
        TransformerDecoderBlockLayerConfig,
        TransformerDecoderLayerConfig,
        TransformerEncoderBlockLayerConfig,
        TransformerEncoderLayerConfig,
    )
    from emperor.transformer._feed_forward import FeedForward, FeedForwardConfig
    from emperor.transformer._layers import (
        TransformerDecoderBlockLayer,
        TransformerDecoderLayer,
        TransformerEncoderBlockLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer._model import Transformer
    from emperor.transformer._state import TransformerDecoderLayerState

__all__ = (
    "Transformer",
    "TransformerConfig",
    "TransformerEncoderLayer",
    "TransformerEncoderBlockLayer",
    "TransformerDecoderBlockLayer",
    "TransformerDecoderLayerState",
    "TransformerDecoderLayer",
    "TransformerEncoderLayerConfig",
    "TransformerEncoderBlockLayerConfig",
    "TransformerDecoderBlockLayerConfig",
    "TransformerDecoderLayerConfig",
    "FeedForward",
    "FeedForwardConfig",
)

_LAZY_EXPORTS = {
    "Transformer": ("emperor.transformer._model", "Transformer"),
    "TransformerConfig": ("emperor.transformer._config", "TransformerConfig"),
    "TransformerEncoderLayer": (
        "emperor.transformer._layers",
        "TransformerEncoderLayer",
    ),
    "TransformerEncoderBlockLayer": (
        "emperor.transformer._layers",
        "TransformerEncoderBlockLayer",
    ),
    "TransformerDecoderBlockLayer": (
        "emperor.transformer._layers",
        "TransformerDecoderBlockLayer",
    ),
    "TransformerDecoderLayerState": (
        "emperor.transformer._state",
        "TransformerDecoderLayerState",
    ),
    "TransformerDecoderLayer": (
        "emperor.transformer._layers",
        "TransformerDecoderLayer",
    ),
    "TransformerEncoderLayerConfig": (
        "emperor.transformer._config",
        "TransformerEncoderLayerConfig",
    ),
    "TransformerEncoderBlockLayerConfig": (
        "emperor.transformer._config",
        "TransformerEncoderBlockLayerConfig",
    ),
    "TransformerDecoderBlockLayerConfig": (
        "emperor.transformer._config",
        "TransformerDecoderBlockLayerConfig",
    ),
    "TransformerDecoderLayerConfig": (
        "emperor.transformer._config",
        "TransformerDecoderLayerConfig",
    ),
    "FeedForward": ("emperor.transformer._feed_forward", "FeedForward"),
    "FeedForwardConfig": (
        "emperor.transformer._feed_forward",
        "FeedForwardConfig",
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
