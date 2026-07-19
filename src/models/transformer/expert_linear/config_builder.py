from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._building import build_experiment_config
from ._validation import validate_runtime
from .runtime_defaults import DEFAULT_RUNTIME, runtime_from_flat
from .runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class TransformerExpertLinearConfigBuilder:
    def __init__(
        self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME, **options: Any
    ) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError("runtime must be a RuntimeOptions value")
        self.runtime = runtime_from_flat(options, runtime) if options else runtime
        validate_runtime(self.runtime)

    def build(self) -> ModelConfig:
        from emperor.config import ModelConfig

        runtime = self.runtime
        return ModelConfig(
            batch_size=runtime.batch_size,
            learning_rate=runtime.learning_rate,
            sequence_length=max(
                runtime.source_sequence_length, runtime.target_sequence_length
            ),
            input_dim=runtime.vocab_size,
            hidden_dim=runtime.model_dim,
            output_dim=runtime.vocab_size,
            experiment_config=build_experiment_config(runtime),
        )


__all__ = ["TransformerExpertLinearConfigBuilder"]
