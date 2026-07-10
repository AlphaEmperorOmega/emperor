"""Compatibility imports for the public Runtime Options Interface."""

from models.bert.linear.runtime_options import (
    SubmoduleStackSource,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)

LEGACY_RUNTIME_OPTIONS_MODULE = True

__all__ = [
    "SubmoduleStackSource",
    "SubmoduleStackOptions",
    "resolve_controller_stack_options",
]
