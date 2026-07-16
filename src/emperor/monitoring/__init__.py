"""Public monitoring Interface shared by Emperor feature packages."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.monitoring._emission import MonitorEmissionPolicy
    from emperor.monitoring._history import MonitorTensorHistory
    from emperor.monitoring._metadata import MonitorOption, MonitorSettings

__all__ = (
    "MonitorOption",
    "MonitorSettings",
    "MonitorEmissionPolicy",
    "MonitorTensorHistory",
)

_LAZY_EXPORTS = {
    "MonitorOption": ("emperor.monitoring._metadata", "MonitorOption"),
    "MonitorSettings": ("emperor.monitoring._metadata", "MonitorSettings"),
    "MonitorEmissionPolicy": (
        "emperor.monitoring._emission",
        "MonitorEmissionPolicy",
    ),
    "MonitorTensorHistory": (
        "emperor.monitoring._history",
        "MonitorTensorHistory",
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
