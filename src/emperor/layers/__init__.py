"""Public Interface for generic layer composition and execution."""

from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    InnerThinkingRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.variants.standard import RecurrentLayer
from emperor.layers._composition.residual.config import (
    AdditiveResidualConfig,
    AttentionResidualConfig,
    ResidualConfig,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._config import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    MirroredLayerStackConfig,
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
    NormalizationOptions,
)
from emperor.layers._stack import LayerStack, MirroredLayerStack
from emperor.layers._state import LayerState

__all__ = (
    "ActivationOptions",
    "AdditiveResidualConfig",
    "AttentionResidualConfig",
    "GateConfig",
    "LastLayerBiasOptions",
    "LayerConfig",
    "LayerGateOptions",
    "LayerNormPositionOptions",
    "NormalizationOptions",
    "LayerStackConfig",
    "MirroredLayerStackConfig",
    "HierarchicalReasoningModelRecurrentConfig",
    "InnerThinkingRecurrentConfig",
    "RecurrentCompositionConfig",
    "RecurrentLayerConfig",
    "TinyRecursiveModelRecurrentConfig",
    "ResidualConfig",
    "WeightedBlendResidualConfig",
    "WeightedResidualConfig",
    "LayerState",
    "Layer",
    "LayerStack",
    "MirroredLayerStack",
    "RecurrentLayer",
    "LayerControllerMonitorCallback",
    "RecurrentLayerMonitorCallback",
)
