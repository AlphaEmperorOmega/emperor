from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.linears import LinearLayerConfig
from models.linears.linear.runtime_options import RuntimeOptions


class ProjectionConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime

    def build_input_model_config(self) -> LayerConfig:
        return LayerConfig(
            activation=self.runtime.stack.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
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
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        )
