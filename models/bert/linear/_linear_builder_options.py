"""Compatibility imports for the public Runtime Options Interface."""

from models.bert.linear.runtime_options import (
    MainLayerStackOptions,
    LayerControllerOptions,
    DynamicMemoryOptions,
    RecurrentControllerOptions,
)

LEGACY_RUNTIME_OPTIONS_MODULE = True

__all__ = [
    "MainLayerStackOptions",
    "LayerControllerOptions",
    "DynamicMemoryOptions",
    "RecurrentControllerOptions",
]
