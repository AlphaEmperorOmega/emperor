from Emperor.config import ModelConfig
from Emperor.base.layer import LayerStackConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.behaviours import DynamicDiagonalOptions
from Emperor.linears.utils.layers import LinearLayerConfig, DynamicLinearLayerConfig
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
)


class LinearsConfigs:
    @staticmethod
    def base_preset(
        batch_size=8,
        input_dim=12,
        output_dim=6,
        bias_flag=True,
    ) -> "ModelConfig":
        return ModelConfig(
            batch_size=batch_size,
            linear_layer_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=DataMonitor,
                parameter_monitor=ParameterMonitor,
            ),
        )

    @staticmethod
    def dynamic_preset(
        batch_size: int = 8,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generators_depth: DynamicDepthOptions = DynamicDepthOptions.DEFAULT,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DEFAULT,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DEFAULT,
        stack_depth: int = 2,
        stack_hidden_dim: int = 0,
    ) -> "ModelConfig":
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else input_dim

        return ModelConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            batch_size=batch_size,
            linear_layer_config=DynamicLinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                generator_depth=generators_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
            ),
            layer_stack_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=stack_hidden_dim,
                output_dim=output_dim,
                num_layers=stack_depth,
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DEFAULT,
                model_type=LinearLayerOptions.BASE,
            ),
        )
