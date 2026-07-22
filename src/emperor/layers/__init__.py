"""Public Interface for generic layer composition and execution."""

from emperor.layers._composition.residual import ResidualConnection
from emperor.layers._config import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    MirroredLayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.layers._layer import Layer
from emperor.layers._mirrored import MirroredLayerStack
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
    "MirroredLayerStackConfig",
    "RecurrentLayerConfig",
    "ResidualConfig",
    "ResidualConnectionOptions",
    "LayerState",
    "ResidualConnection",
    "Layer",
    "LayerStack",
    "MirroredLayerStack",
    "RecurrentLayer",
    "LayerControllerMonitorCallback",
    "RecurrentLayerMonitorCallback",
)
