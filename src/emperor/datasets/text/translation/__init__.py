"""Public Interface for supported Multi30k translation adapters."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.datasets.text.translation._adapter import (
        Multi30kDeEn,
        Multi30kEnDe,
    )

__all__ = ("Multi30kDeEn", "Multi30kEnDe")

_LAZY_EXPORTS = {
    "Multi30kDeEn": (
        "emperor.datasets.text.translation._adapter",
        "Multi30kDeEn",
    ),
    "Multi30kEnDe": (
        "emperor.datasets.text.translation._adapter",
        "Multi30kEnDe",
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
