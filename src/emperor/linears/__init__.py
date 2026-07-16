"""Public Interface for generic linear layers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.linears._config import LinearLayerConfig
    from emperor.linears._layer import LinearAbstract, LinearLayer
    from emperor.linears._monitoring.callback import LinearMonitorCallback
    from emperor.linears._options import LinearOptions

__all__ = (
    "LinearLayer",
    "LinearAbstract",
    "LinearLayerConfig",
    "LinearOptions",
    "LinearMonitorCallback",
)

_LAZY_EXPORTS = {
    "LinearLayer": ("emperor.linears._layer", "LinearLayer"),
    "LinearAbstract": ("emperor.linears._layer", "LinearAbstract"),
    "LinearLayerConfig": ("emperor.linears._config", "LinearLayerConfig"),
    "LinearOptions": ("emperor.linears._options", "LinearOptions"),
    "LinearMonitorCallback": (
        "emperor.linears._monitoring.callback",
        "LinearMonitorCallback",
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
