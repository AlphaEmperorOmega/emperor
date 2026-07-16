"""Public convolution Interface."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.convs._config import Conv2dLayerConfig
    from emperor.convs._layer import Conv2dLayer

__all__ = ("Conv2dLayerConfig", "Conv2dLayer")

_LAZY_EXPORTS = {
    "Conv2dLayerConfig": ("emperor.convs._config", "Conv2dLayerConfig"),
    "Conv2dLayer": ("emperor.convs._layer", "Conv2dLayer"),
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
