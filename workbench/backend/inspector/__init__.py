"""Public inspector helpers with lazy imports.

The API app can expose snapshot routes without importing the full model
inspection stack. Keep heavyweight discovery/model imports behind attribute
access so lightweight routes and tests do not require ML runtime dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ModelParts",
    "config_schema",
    "discover_models",
    "inspect_model",
    "list_model_presets",
    "load_model_parts",
    "serialize_graph",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ModelParts",
        "discover_models",
        "list_model_presets",
        "load_model_parts",
    }:
        from workbench.backend.inspector import discovery

        return getattr(discovery, name)
    if name == "serialize_graph":
        from workbench.backend.inspector.graph import serialize_graph

        return serialize_graph
    if name == "config_schema":
        from workbench.backend.inspector.schema import config_schema

        return config_schema
    if name == "inspect_model":
        from workbench.backend.inspector.service import inspect_model

        return inspect_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
