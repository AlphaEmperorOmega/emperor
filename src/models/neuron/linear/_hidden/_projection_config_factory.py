from emperor.base.layer.config import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import LinearLayerConfig

from models.neuron.linear._hidden.runtime_options import RuntimeOptions


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime

    def build_input_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=self.runtime.stack.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )

    def build_output_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )
