from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ._runtime_defaults_resolver import (
    _NeuronExpertLinearAdaptiveRuntimeDefaultsResolver,
)
from .runtime_options import RuntimeOptions


def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    resolver = _NeuronExpertLinearAdaptiveRuntimeDefaultsResolver(**dict(values or {}))
    return RuntimeOptions(
        {
            "hidden_runtime": resolver.hidden_runtime,
            "cluster_capacity_options": resolver.cluster_capacity_options,
            "terminal_options": resolver.terminal_options,
            "terminal_router_options": resolver.terminal_router_options,
            "terminal_sampler_options": resolver.terminal_sampler_options,
            "cluster_halting_options": resolver.cluster_halting_options,
        }
    )


DEFAULT_RUNTIME: Final = runtime_from_flat()

__all__ = ["DEFAULT_RUNTIME", "runtime_from_flat"]
