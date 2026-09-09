from emperor.config import ModelConfig
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    NormalizationOptions,
)
from emperor.linears import LinearLayerConfig
from models.parametric.parametric_vector._control_config_factory import (
    build_parametric_stack_config,
)
from models.parametric.parametric_vector._runtime_construction import (
    ParametricVectorConstructionOptions,
)
from models.parametric.parametric_vector.experiment_config import ExperimentConfig
from models.parametric.parametric_vector.runtime_defaults import DEFAULT_RUNTIME
from models.parametric.parametric_vector.runtime_options import RuntimeOptions


class ParametricVectorConfigBuilder:
    """Build a vector ModelConfig from one resolved package Parameter Object."""

    def __init__(self, *, runtime: RuntimeOptions = DEFAULT_RUNTIME) -> None:
        if type(runtime) is not RuntimeOptions:
            raise TypeError(
                "models.parametric.parametric_vector "
                "ParametricVectorConfigBuilder runtime must be RuntimeOptions"
            )
        self.runtime = runtime
        self._options: ParametricVectorConstructionOptions = (
            runtime.construction_options()
        )

    def build(self) -> ModelConfig:
        options = self._options
        input_model_config = build_linear_layer_config(
            activation=options.stack.activation,
        )
        model_config = build_parametric_stack_config(
            input_dim=options.hidden_dim,
            hidden_dim=options.hidden_dim,
            output_dim=options.hidden_dim,
            stack_options=options.stack,
            mixture_options=options.mixture,
            sampler_options=options.sampler,
            router_options=options.router,
            residual_stack_options=options.residual_stack,
        )
        output_model_config = build_linear_layer_config(
            activation=ActivationOptions.DISABLED,
        )
        return ModelConfig(
            batch_size=options.batch_size,
            input_dim=options.input_dim,
            learning_rate=options.learning_rate,
            hidden_dim=options.hidden_dim,
            output_dim=options.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )


def build_linear_layer_config(
    *,
    activation: ActivationOptions,
) -> LayerConfig:
    layer_model_config = LinearLayerConfig(
        bias_flag=True,
    )
    return LayerConfig(
        activation=activation,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        normalization=NormalizationOptions.RMS_NORM,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )


__all__ = ["ParametricVectorConfigBuilder", "build_linear_layer_config"]
