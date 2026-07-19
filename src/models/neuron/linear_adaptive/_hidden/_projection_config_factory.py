from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig

from models.neuron.linear_adaptive._hidden._adaptive_parameter_config_factory import (
    AdaptiveParameterConfigFactory,
)
from models.neuron.linear_adaptive._hidden.runtime_options import (
    AdaptiveProjectionOptions,
    RuntimeOptions,
)


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime
        self._adaptive_factory = AdaptiveParameterConfigFactory(runtime)

    def build_input_model_config(self) -> LayerConfig:
        return self._build(
            self._runtime.input_projection,
            activation=self._runtime.stack.activation,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self._build(
            self._runtime.output_projection,
            activation=ActivationOptions.DISABLED,
        )

    def _build(
        self,
        options: AdaptiveProjectionOptions,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=self._runtime.stack.bias_flag,
                adaptive_augmentation_config=(
                    self._adaptive_factory.build_projection_config(options)
                ),
            ),
        )
