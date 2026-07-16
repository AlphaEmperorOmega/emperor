"""Public Interface for input-dependent parametric modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from emperor.parametric._mixtures.matrix import (
        MatrixBiasMixture,
        MatrixWeightsMixture,
    )
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

_LAZY_EXPORTS = {
    "AdaptiveRouterOptions": (
        "emperor.parametric._config",
        "AdaptiveRouterOptions",
    ),
    "ParametricLayerConfig": (
        "emperor.parametric._config",
        "ParametricLayerConfig",
    ),
    "ParametricLayerHandlerConfig": (
        "emperor.parametric._config",
        "ParametricLayerHandlerConfig",
    ),
    "ParametricLayer": (
        "emperor.parametric._layer",
        "ParametricLayer",
    ),
    "ParametricLayerMonitorCallback": (
        "emperor.parametric._monitoring",
        "ParametricLayerMonitorCallback",
    ),
    "ParametricLayerHandler": (
        "emperor.parametric._handlers",
        "ParametricLayerHandler",
    ),
    "ParametricLayerState": (
        "emperor.parametric._state",
        "ParametricLayerState",
    ),
    "AdaptiveMixtureBase": (
        "emperor.parametric._mixtures.base",
        "AdaptiveMixtureBase",
    ),
    "AdaptiveMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "AdaptiveMixtureConfig",
    ),
    "ClipParameterOptions": (
        "emperor.parametric._mixtures.config",
        "ClipParameterOptions",
    ),
    "VectorWeightsMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "VectorWeightsMixtureConfig",
    ),
    "MatrixWeightsMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "MatrixWeightsMixtureConfig",
    ),
    "MatrixBiasMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "MatrixBiasMixtureConfig",
    ),
    "GeneratorWeightsMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "GeneratorWeightsMixtureConfig",
    ),
    "GeneratorBiasMixtureConfig": (
        "emperor.parametric._mixtures.config",
        "GeneratorBiasMixtureConfig",
    ),
    "VectorWeightsMixture": (
        "emperor.parametric._mixtures.vector",
        "VectorWeightsMixture",
    ),
    "MatrixWeightsMixture": (
        "emperor.parametric._mixtures.matrix",
        "MatrixWeightsMixture",
    ),
    "MatrixBiasMixture": (
        "emperor.parametric._mixtures.matrix",
        "MatrixBiasMixture",
    ),
    "GeneratorWeightsMixture": (
        "emperor.parametric._mixtures.generator",
        "GeneratorWeightsMixture",
    ),
    "GeneratorBiasMixture": (
        "emperor.parametric._mixtures.generator",
        "GeneratorBiasMixture",
    ),
    "VectorRouterConfig": (
        "emperor.parametric._routing",
        "VectorRouterConfig",
    ),
    "VectorRouterModel": (
        "emperor.parametric._routing",
        "VectorRouterModel",
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
