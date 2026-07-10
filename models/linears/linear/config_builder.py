from typing import TYPE_CHECKING

from models.linears.linear._hidden_model_config_factory import HiddenModelConfigFactory
from models.linears.linear._projection_config_factory import ProjectionConfigFactory
from models.linears.linear.experiment_config import ExperimentConfig
from models.linears.linear.runtime_defaults import DEFAULT_RUNTIME
from models.linears.linear.runtime_options import RuntimeOptions

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearConfigBuilder:
    def __init__(
        self,
        *,
        runtime: RuntimeOptions = DEFAULT_RUNTIME,
    ) -> None:
        if type(runtime) is not RuntimeOptions:
            actual_type = type(runtime)
            raise TypeError(
                "models.linears.linear LinearConfigBuilder 'runtime' must be "
                "models.linears.linear.runtime_options.RuntimeOptions; got "
                f"{actual_type.__module__}.{actual_type.__qualname__}"
            )
        self.runtime = runtime

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        projection_factory = ProjectionConfigFactory(self.runtime)
        hidden_factory = HiddenModelConfigFactory(self.runtime)
        return ModelConfig(
            learning_rate=self.runtime.learning_rate,
            batch_size=self.runtime.batch_size,
            input_dim=self.runtime.input_dim,
            hidden_dim=self.runtime.hidden_dim,
            output_dim=self.runtime.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=projection_factory.build_input_model_config(),
                model_config=hidden_factory.build_hidden_model_config(),
                output_model_config=projection_factory.build_output_model_config(),
            ),
        )
