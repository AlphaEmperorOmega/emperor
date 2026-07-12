"""Compatibility Adapter for the Emperor semantic model graph."""

from __future__ import annotations

from emperor.inspection import (
    ARCHITECTURE_ROLE,
    INTERNAL_ROLE,
    ROOT_NODE_ID,
    ROOT_NODE_PATH,
    RUNTIME_ROLE,
    GraphRole,
    graph_role,
    module_details,
    parameter_count,
    parameter_size_bytes,
)
from torch.nn import Module

from workbench.backend.inspection_adapter import inspect_module_graph_payload


def serialize_graph(module: Module):
    return inspect_module_graph_payload(module)


__all__ = [
    "ARCHITECTURE_ROLE",
    "GraphRole",
    "INTERNAL_ROLE",
    "ROOT_NODE_ID",
    "ROOT_NODE_PATH",
    "RUNTIME_ROLE",
    "graph_role",
    "module_details",
    "parameter_count",
    "parameter_size_bytes",
    "serialize_graph",
]
