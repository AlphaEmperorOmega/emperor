from Emperor.config import ModelConfig
from Emperor.base.utils import ConfigUtils
from Emperor.base.layer import LayerStackConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.behaviours.utils.behaviours import DynamicDiagonalOptions
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.linears.options import LinearLayerOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class LinearsConfigs:
    @staticmethod
    def linear_layer_config(
        input_dim=12,
        output_dim=6,
        bias_flag=True,
        data_monitor=None,
        parameter_monitor=None,
        **kwargs,
    ) -> "LinearLayerConfig":
        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=data_monitor,
            parameter_monitor=parameter_monitor,
        )

    @staticmethod
    def adaptive_linear_layer_config(
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        **kwargs,
    ):
        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=None,
            parameter_monitor=None,
            generator_depth=generator_depth,
            diagonal_option=diagonal_option,
            bias_option=bias_option,
            memory_option=memory_option,
            memory_size_option=memory_size_option,
            memory_position_option=memory_position_option,
            override_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=stack_hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.NONE,
                residual_flag=residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=dropout_probability,
                override_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    generator_depth=generator_depth,
                    data_monitor=None,
                    parameter_monitor=None,
                ),
            ),
        )

    @staticmethod
    def base_preset(
        batch_size=8,
        input_dim=12,
        output_dim=6,
        bias_flag=True,
        data_monitor=None,
        parameter_monitor=None,
    ) -> "ModelConfig":
        arguments = ConfigUtils.get_method_arguments()
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            linear_layer_config=LinearsConfigs.linear_layer_config(**arguments),
        )

    @staticmethod
    def dynamic_preset(
        batch_size: int = 8,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else input_dim
        arguments = ConfigUtils.get_method_arguments()
        return ModelConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            batch_size=batch_size,
            linear_layer_config=LinearsConfigs.adaptive_linear_layer_config(
                **arguments
            ),
            layer_stack_config=LayerStackConfig(
                model_type=LinearLayerOptions.BASE,
                input_dim=input_dim,
                hidden_dim=stack_hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.NONE,
                residual_flag=residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=dropout_probability,
            ),
        )
