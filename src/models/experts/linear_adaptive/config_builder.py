from __future__ import annotations

from models.experts.linear_adaptive._config_implementation import (
    _RuntimeDefaultsResolver,
)
from models.experts.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.experts.linear_adaptive.runtime_options import RuntimeOptions


class LinearAdaptiveConfigBuilder(_RuntimeDefaultsResolver):
    """Build adaptive-expert configs from one immutable runtime value."""

    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")

        self.runtime = runtime
        self.__dict__.update(runtime._resolved_state)


__all__ = ["LinearAdaptiveConfigBuilder"]
