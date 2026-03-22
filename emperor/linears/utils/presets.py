from emperor.base.layer import LayerStackConfig
from emperor.linears.options import LinearLayerOptions
from emperor.linears.utils.config import LinearLayerConfig
from emperor.behaviours.model import AdaptiveParameterBehaviourConfig
from emperor.behaviours.options import DynamicDiagonalOptions
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.behaviours.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
    WeightNormalizationOptions,
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
        from emperor.config import ModelConfig

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
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.NONE,
        weight_option: DynamicWeightOptions = DynamicWeightOptions.DUAL_MODEL,
        weight_normalization: WeightNormalizationOptions = WeightNormalizationOptions.CLAMP,
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
        from emperor.config import ModelConfig

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
                weight_option=weight_option,
                weight_normalization=weight_normalization,
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
                    layer_norm_position=layer_norm_position,
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
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.NONE,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
    ) -> "LayerStackConfig":
        from emperor.config import ModelConfig

        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LayerStackConfig(
            model_type=LinearLayerOptions.BASE,
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
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
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.NONE,
        weight_option: DynamicWeightOptions = DynamicWeightOptions.DUAL_MODEL,
        weight_normalization: WeightNormalizationOptions = WeightNormalizationOptions.CLAMP,
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
        adaptive_behaviour_stack_num_layers: int = 2,
    ) -> "LayerStackConfig":
        from emperor.config import ModelConfig

        _hidden_dim = max(input_dim, output_dim)
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else _hidden_dim

        config = LayerStackConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
            input_dim=input_dim,
            hidden_dim=stack_hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            activation=stack_activation,
            layer_norm_position=layer_norm_position,
            residual_flag=stack_residual_flag,
            adaptive_computation_flag=False,
            dropout_probability=stack_dropout_probability,
            override_config=LinearPresets.adaptive_linear_layer_preset(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                weight_option=weight_option,
                weight_normalization=weight_normalization,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
                stack_num_layers=adaptive_behaviour_stack_num_layers,
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
