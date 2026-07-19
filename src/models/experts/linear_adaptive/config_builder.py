from __future__ import annotations

from typing import Any

from models.experts.linear_adaptive._config_implementation import (
    _LegacyLinearAdaptiveConfigResolver,
)
from models.experts.linear_adaptive.runtime_defaults import (
    DEFAULT_RUNTIME,
    runtime_from_legacy_options,
)
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


class LinearAdaptiveConfigBuilder(_LegacyLinearAdaptiveConfigResolver):
    """Build adaptive-expert configs from one immutable runtime value.

    Positional and keyword legacy construction remains accepted while callers
    migrate to ``runtime``. Legacy arguments are resolved immediately into the
    same package-local Runtime Options used by presets and search.
    """

    def __init__(
        self,
        *legacy_args: Any,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
        **legacy_options: Any,
    ) -> None:
        if legacy_args or legacy_options:
            if runtime is not DEFAULT_RUNTIME:
                raise TypeError(
                    "runtime cannot be combined with legacy builder options"
                )
            runtime = runtime_from_legacy_options(*legacy_args, **legacy_options)
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")

        self.runtime = runtime
        self.__dict__.update(runtime._resolved_state)


__all__ = ["LinearAdaptiveConfigBuilder"]
