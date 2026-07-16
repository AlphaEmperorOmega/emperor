"""Parameter augmentation family Modules."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations import adaptive_parameters

__all__ = ("adaptive_parameters",)


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)

    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value
