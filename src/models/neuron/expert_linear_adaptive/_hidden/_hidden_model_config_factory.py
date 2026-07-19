from __future__ import annotations

from models.neuron.expert_linear_adaptive._hidden._config_implementation import (
    _LegacyLinearAdaptiveConfigResolver,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import RuntimeOptions


class HiddenModelConfigFactory(_LegacyLinearAdaptiveConfigResolver):
    """Build package-local boundaries and hidden expert-adaptive configuration."""

    def __init__(self, runtime: RuntimeOptions) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime
        self.__dict__.update(runtime._resolved_state)

    def build_input_model_config(self):
        return self._build_input_model_config()

    def build_hidden_model_config(self):
        return self._build_hidden_model_config()

    def build_output_model_config(self):
        return self._build_output_model_config()


__all__ = ["HiddenModelConfigFactory"]
