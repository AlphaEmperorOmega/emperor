"""Public Interface for mixture-of-experts modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.experts._config import (
        MixtureOfExpertsConfig,
        MixtureOfExpertsLayerConfig,
        MixtureOfExpertsModelConfig,
    )
    from emperor.experts._layers.map import MixtureOfExpertsMap
    from emperor.experts._layers.mixture import MixtureOfExperts
    from emperor.experts._layers.reduce import MixtureOfExpertsReduce
    from emperor.experts._model import MixtureOfExpertsModel
    from emperor.experts._options import (
        DroppedTokenOptions,
        ExpertWeightingPositionOptions,
        RoutingInitializationMode,
    )
    from emperor.experts._state import MixtureOfExpertsLayerState

__all__ = (
    "DroppedTokenOptions",
    "ExpertWeightingPositionOptions",
    "RoutingInitializationMode",
    "MixtureOfExperts",
    "MixtureOfExpertsConfig",
    "MixtureOfExpertsLayerConfig",
    "MixtureOfExpertsLayerState",
    "MixtureOfExpertsMap",
    "MixtureOfExpertsModel",
    "MixtureOfExpertsModelConfig",
    "MixtureOfExpertsReduce",
)

_LAZY_EXPORTS = {
    "DroppedTokenOptions": (
        "emperor.experts._options",
        "DroppedTokenOptions",
    ),
    "ExpertWeightingPositionOptions": (
        "emperor.experts._options",
        "ExpertWeightingPositionOptions",
    ),
    "RoutingInitializationMode": (
        "emperor.experts._options",
        "RoutingInitializationMode",
    ),
    "MixtureOfExperts": (
        "emperor.experts._layers.mixture",
        "MixtureOfExperts",
    ),
    "MixtureOfExpertsConfig": (
        "emperor.experts._config",
        "MixtureOfExpertsConfig",
    ),
    "MixtureOfExpertsLayerConfig": (
        "emperor.experts._config",
        "MixtureOfExpertsLayerConfig",
    ),
    "MixtureOfExpertsLayerState": (
        "emperor.experts._state",
        "MixtureOfExpertsLayerState",
    ),
    "MixtureOfExpertsMap": (
        "emperor.experts._layers.map",
        "MixtureOfExpertsMap",
    ),
    "MixtureOfExpertsModel": (
        "emperor.experts._model",
        "MixtureOfExpertsModel",
    ),
    "MixtureOfExpertsModelConfig": (
        "emperor.experts._config",
        "MixtureOfExpertsModelConfig",
    ),
    "MixtureOfExpertsReduce": (
        "emperor.experts._layers.reduce",
        "MixtureOfExpertsReduce",
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
