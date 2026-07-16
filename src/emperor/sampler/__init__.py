"""Public Interface for routing and expert selection."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.sampler._config import RouterConfig, SamplerConfig
    from emperor.sampler._monitoring import SamplerMonitorCallback
    from emperor.sampler._router import RouterModel
    from emperor.sampler._sampler import SamplerModel

__all__ = (
    "RouterConfig",
    "SamplerConfig",
    "RouterModel",
    "SamplerModel",
    "SamplerMonitorCallback",
)

_LAZY_EXPORTS = {
    "RouterConfig": ("emperor.sampler._config", "RouterConfig"),
    "SamplerConfig": ("emperor.sampler._config", "SamplerConfig"),
    "RouterModel": ("emperor.sampler._router", "RouterModel"),
    "SamplerModel": ("emperor.sampler._sampler", "SamplerModel"),
    "SamplerMonitorCallback": (
        "emperor.sampler._monitoring",
        "SamplerMonitorCallback",
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
