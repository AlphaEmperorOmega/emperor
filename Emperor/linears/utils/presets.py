from Emperor.config import ModelConfig
from Emperor.base.layer import LayerStackConfig
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from Emperor.behaviours.utils.behaviours import DynamicDiagonalOptions
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class LinearPresets:
    @staticmethod
    def base_linear_layer_preset(
        return_model_config_flag: bool = False,
        batch_size: int = 8,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        data_monitor=None,
        parameter_monitor=None,
    ) -> "LinearLayerConfig | ModelConfig":
        config = LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=data_monitor,
            parameter_monitor=parameter_monitor,
        )
        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            linear_layer_config=config,
        )

    @staticmethod
    def adaptive_linear_layer_preset(
        return_model_config_flag: bool = False,
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
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "LinearLayerConfig | ModelConfig":
        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=None,
            parameter_monitor=None,
            override_config=AdaptiveParameterBehaviourConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                override_config=LayerStackConfig(
                    model_type=LinearLayerOptions.BASE,
                    input_dim=input_dim,
                    hidden_dim=stack_hidden_dim,
                    output_dim=output_dim,
                    num_layers=stack_num_layers,
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=stack_residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=stack_dropout_probability,
                    override_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        data_monitor=None,
                        parameter_monitor=None,
                        override_config=AdaptiveParameterBehaviourConfig(
                            generator_depth=generator_depth,
                        ),
                    ),
                ),
            ),
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            linear_layer_config=config,
        )

    @staticmethod
    def base_linear_layer_stack_preset(
        return_model_config_flag: bool = False,
        batch_size: int = 8,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        data_monitor=None,
        parameter_monitor=None,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig | ModelConfig":
        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=LayerNormPositionOptions.NONE,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=LinearPresets.base_linear_layer_preset(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=data_monitor,
                parameter_monitor=parameter_monitor,
            ),
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_stack_config=config,
        )

    @staticmethod
    def adaptive_linear_layer_stack_preset(
        return_model_config_flag: bool = False,
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
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig | ModelConfig":
        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LayerStackConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=LayerNormPositionOptions.NONE,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=LinearPresets.adaptive_linear_layer_preset(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_flag=stack_residual_flag,
                stack_dropout_probability=stack_dropout_probability,
            ),
        )

        if not return_model_config_flag:
            return config
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            layer_stack_config=config,
        )
