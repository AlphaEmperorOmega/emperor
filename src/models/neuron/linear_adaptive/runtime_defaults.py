from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ._runtime_defaults_resolver import _NeuronLinearAdaptiveRuntimeDefaultsResolver
from .runtime_options import RuntimeOptions

_UNKNOWN_KEY_MARKERS = (
    "unknown runtime override",
    "unknown runtime key",
    "unexpected hidden runtime option",
    "got an unexpected keyword argument",
)


def _keyword_values(**values: object) -> dict[str, object]:
    return values


def _raise_outer_unknown_key(
    values: Mapping[str, object],
    error: TypeError | ValueError,
) -> None:
    message = str(error)
    if not any(marker in message for marker in _UNKNOWN_KEY_MARKERS):
        raise error
    for key in values:
        if repr(key) in message:
            raise TypeError(
                f"_NeuronLinearAdaptiveRuntimeDefaultsResolver.__init__() got an unexpected keyword argument {key!r}"
            ) from None
    raise error


def runtime_from_flat(values: Mapping[str, object] | None = None) -> RuntimeOptions:
    flat_values = _keyword_values(**dict(values or {}))
    try:
        resolver = _NeuronLinearAdaptiveRuntimeDefaultsResolver(flat_values)
    except (TypeError, ValueError) as error:
        _raise_outer_unknown_key(flat_values, error)
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
