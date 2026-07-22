from __future__ import annotations

from emperor.config import ModelConfig

from ._building import encoder_config, output_config, patch_config, sequence_length
from .experiment_config import ExperimentConfig
from .runtime_defaults import DEFAULT_RUNTIME
from .runtime_options import RuntimeOptions


class MlpMixerLinearAdaptiveConfigBuilder:
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime
        self.batch_size = runtime.batch_size
        self.learning_rate = runtime.learning_rate
        self.input_dim = runtime.input_dim
        self.hidden_dim = runtime.hidden_dim
        self.output_dim = runtime.output_dim
        self.sequence_length = sequence_length(runtime)

    def build(self) -> ModelConfig:
        runtime = self.runtime
        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            experiment_config=ExperimentConfig(
                patch_config=patch_config(runtime),
                encoder_config=encoder_config(runtime, self.sequence_length),
                output_config=output_config(runtime),
            ),
        )


__all__ = ["MlpMixerLinearAdaptiveConfigBuilder"]
