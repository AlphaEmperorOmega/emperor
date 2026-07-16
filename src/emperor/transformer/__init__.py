"""Public Interface for Transformer modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.transformer._config import (
        TransformerConfig,
        TransformerDecoderBlockLayerConfig,
        TransformerDecoderLayerConfig,
        TransformerDecoderStackConfig,
        TransformerEncoderBlockLayerConfig,
        TransformerEncoderLayerConfig,
        TransformerEncoderStackConfig,
    )
    from emperor.transformer._feed_forward import FeedForward, FeedForwardConfig
    from emperor.transformer._layers import (
        TransformerDecoderBlockLayer,
        TransformerDecoderLayer,
        TransformerEncoderBlockLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer._model import Transformer
    from emperor.transformer._options.config_adapter import (
        attention_options_from_config,
        feed_forward_options_from_config,
    )
    from emperor.transformer._options.overrides import (
        expand_transformer_path_locks,
        resolve_transformer_path_options,
    )
    from emperor.transformer._options.records import (
        ControllerStackOptions,
        DynamicMemoryOptions,
        LayerControllerOptions,
        RecurrentControllerOptions,
        SubmoduleStackOptions,
        SubmoduleStackSource,
        TransformerAttentionOptions,
        TransformerFeedForwardOptions,
        TransformerPathOptions,
        TransformerStackOptions,
        resolve_controller_stack_options,
    )
    from emperor.transformer._stacks import (
        TransformerDecoderStack,
        TransformerEncoderStack,
    )
    from emperor.transformer._state import TransformerDecoderLayerState
    from emperor.transformer._submodule_configuration import (
        configure_transformer_submodule,
    )

__all__ = (
    "Transformer",
    "TransformerConfig",
    "TransformerEncoderStack",
    "TransformerDecoderStack",
    "TransformerEncoderStackConfig",
    "TransformerDecoderStackConfig",
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
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "SubmoduleStackOptions",
    "SubmoduleStackSource",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerPathOptions",
    "TransformerStackOptions",
    "attention_options_from_config",
    "configure_transformer_submodule",
    "expand_transformer_path_locks",
    "feed_forward_options_from_config",
    "resolve_controller_stack_options",
    "resolve_transformer_path_options",
)

_LAZY_EXPORTS = {
    "Transformer": ("emperor.transformer._model", "Transformer"),
    "TransformerConfig": ("emperor.transformer._config", "TransformerConfig"),
    "TransformerEncoderStack": (
        "emperor.transformer._stacks",
        "TransformerEncoderStack",
    ),
    "TransformerDecoderStack": (
        "emperor.transformer._stacks",
        "TransformerDecoderStack",
    ),
    "TransformerEncoderStackConfig": (
        "emperor.transformer._config",
        "TransformerEncoderStackConfig",
    ),
    "TransformerDecoderStackConfig": (
        "emperor.transformer._config",
        "TransformerDecoderStackConfig",
    ),
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
    "ControllerStackOptions": (
        "emperor.transformer._options.records",
        "ControllerStackOptions",
    ),
    "DynamicMemoryOptions": (
        "emperor.transformer._options.records",
        "DynamicMemoryOptions",
    ),
    "LayerControllerOptions": (
        "emperor.transformer._options.records",
        "LayerControllerOptions",
    ),
    "RecurrentControllerOptions": (
        "emperor.transformer._options.records",
        "RecurrentControllerOptions",
    ),
    "SubmoduleStackOptions": (
        "emperor.transformer._options.records",
        "SubmoduleStackOptions",
    ),
    "SubmoduleStackSource": (
        "emperor.transformer._options.records",
        "SubmoduleStackSource",
    ),
    "TransformerAttentionOptions": (
        "emperor.transformer._options.records",
        "TransformerAttentionOptions",
    ),
    "TransformerFeedForwardOptions": (
        "emperor.transformer._options.records",
        "TransformerFeedForwardOptions",
    ),
    "TransformerPathOptions": (
        "emperor.transformer._options.records",
        "TransformerPathOptions",
    ),
    "TransformerStackOptions": (
        "emperor.transformer._options.records",
        "TransformerStackOptions",
    ),
    "attention_options_from_config": (
        "emperor.transformer._options.config_adapter",
        "attention_options_from_config",
    ),
    "configure_transformer_submodule": (
        "emperor.transformer._submodule_configuration",
        "configure_transformer_submodule",
    ),
    "expand_transformer_path_locks": (
        "emperor.transformer._options.overrides",
        "expand_transformer_path_locks",
    ),
    "feed_forward_options_from_config": (
        "emperor.transformer._options.config_adapter",
        "feed_forward_options_from_config",
    ),
    "resolve_controller_stack_options": (
        "emperor.transformer._options.records",
        "resolve_controller_stack_options",
    ),
    "resolve_transformer_path_options": (
        "emperor.transformer._options.overrides",
        "resolve_transformer_path_options",
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
