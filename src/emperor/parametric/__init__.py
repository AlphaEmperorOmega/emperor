"""Public Interface for input-dependent parametric modules."""

from emperor.parametric._config import (
    AdaptiveRouterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from emperor.parametric._handlers import ParametricLayerHandler
from emperor.parametric._layer import ParametricLayer
from emperor.parametric._mixtures.base import AdaptiveMixtureBase
from emperor.parametric._mixtures.config import (
    AdaptiveMixtureConfig,
    ClipParameterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._mixtures.generator import (
    GeneratorBiasMixture,
    GeneratorWeightsMixture,
)
from emperor.parametric._mixtures.matrix import MatrixBiasMixture, MatrixWeightsMixture
from emperor.parametric._mixtures.vector import VectorWeightsMixture
from emperor.parametric._monitoring import ParametricLayerMonitorCallback
from emperor.parametric._routing import VectorRouterConfig, VectorRouterModel
from emperor.parametric._state import ParametricLayerState

__all__ = (
    "AdaptiveRouterOptions",
    "ParametricLayerConfig",
    "ParametricLayerHandlerConfig",
    "ParametricLayer",
    "ParametricLayerMonitorCallback",
    "ParametricLayerHandler",
    "ParametricLayerState",
    "AdaptiveMixtureBase",
    "AdaptiveMixtureConfig",
    "ClipParameterOptions",
    "VectorWeightsMixtureConfig",
    "MatrixWeightsMixtureConfig",
    "MatrixBiasMixtureConfig",
    "GeneratorWeightsMixtureConfig",
    "GeneratorBiasMixtureConfig",
    "VectorWeightsMixture",
    "MatrixWeightsMixture",
    "MatrixBiasMixture",
    "GeneratorWeightsMixture",
    "GeneratorBiasMixture",
    "VectorRouterConfig",
    "VectorRouterModel",
)
