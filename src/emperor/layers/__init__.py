"""Public Interface for generic layer composition and execution."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.layers._composition.residual import ResidualConnection
    from emperor.layers._config import (
        GateConfig,
        LayerConfig,
        LayerStackConfig,
        RecurrentLayerConfig,
    )
    from emperor.layers._layer import Layer
    from emperor.layers._monitoring.callbacks import (
        LayerControllerMonitorCallback,
        RecurrentLayerMonitorCallback,
    )
    from emperor.layers._options import (
        ActivationOptions,
        LastLayerBiasOptions,
        LayerGateOptions,
        LayerNormPositionOptions,
        ResidualConnectionOptions,
    )
    from emperor.layers._recurrent import RecurrentLayer
    from emperor.layers._stack import LayerStack
    from emperor.layers._state import LayerState

__all__ = (
    "ActivationOptions",
    "GateConfig",
    "LastLayerBiasOptions",
    "LayerConfig",
    "LayerGateOptions",
    "LayerNormPositionOptions",
    "LayerStackConfig",
    "RecurrentLayerConfig",
    "ResidualConnectionOptions",
    "LayerState",
    "ResidualConnection",
    "Layer",
    "LayerStack",
    "RecurrentLayer",
    "LayerControllerMonitorCallback",
    "RecurrentLayerMonitorCallback",
)

_LAZY_EXPORTS = {
    "ActivationOptions": ("emperor.layers._options", "ActivationOptions"),
    "GateConfig": ("emperor.layers._config", "GateConfig"),
    "LastLayerBiasOptions": ("emperor.layers._options", "LastLayerBiasOptions"),
    "LayerConfig": ("emperor.layers._config", "LayerConfig"),
    "LayerGateOptions": ("emperor.layers._options", "LayerGateOptions"),
    "LayerNormPositionOptions": (
        "emperor.layers._options",
        "LayerNormPositionOptions",
    ),
    "LayerStackConfig": ("emperor.layers._config", "LayerStackConfig"),
    "RecurrentLayerConfig": ("emperor.layers._config", "RecurrentLayerConfig"),
    "ResidualConnectionOptions": (
        "emperor.layers._options",
        "ResidualConnectionOptions",
    ),
    "LayerState": ("emperor.layers._state", "LayerState"),
    "ResidualConnection": (
        "emperor.layers._composition.residual",
        "ResidualConnection",
    ),
    "Layer": ("emperor.layers._layer", "Layer"),
    "LayerStack": ("emperor.layers._stack", "LayerStack"),
    "RecurrentLayer": ("emperor.layers._recurrent", "RecurrentLayer"),
    "LayerControllerMonitorCallback": (
        "emperor.layers._monitoring.callbacks",
        "LayerControllerMonitorCallback",
    ),
    "RecurrentLayerMonitorCallback": (
        "emperor.layers._monitoring.callbacks",
        "RecurrentLayerMonitorCallback",
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
