from emperor.config import ModelConfig
from models.linears.linear_adaptive._hidden_model_config_factory import (
    HiddenModelConfigFactory,
)
from models.linears.linear_adaptive._projection_config_factory import (
    ProjectionConfigFactory,
)
from models.linears.linear_adaptive.experiment_config import ExperimentConfig
from models.linears.linear_adaptive.runtime_defaults import DEFAULT_RUNTIME
from models.linears.linear_adaptive.runtime_options import RuntimeOptions


class LinearAdaptiveConfigBuilder:
    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if not isinstance(runtime, RuntimeOptions):
            raise TypeError(
                "models.linears.linear_adaptive: runtime must be this package's "
                f"RuntimeOptions, got {type(runtime).__module__}."
                f"{type(runtime).__qualname__}"
            )
        self.runtime = runtime

    def build(self) -> ModelConfig:
        runtime = self.runtime
        projections = ProjectionConfigFactory(runtime)
        return ModelConfig(
            learning_rate=runtime.learning_rate,
            batch_size=runtime.batch_size,
            input_dim=runtime.input_dim,
            hidden_dim=runtime.hidden_dim,
            output_dim=runtime.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=projections.build_input_model_config(),
                model_config=HiddenModelConfigFactory(
                    runtime
                ).build_hidden_model_config(),
                output_model_config=projections.build_output_model_config(),
            ),
        )
