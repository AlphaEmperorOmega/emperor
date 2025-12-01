from Emperor.config import ModelConfig
from Emperor.base.layer import LayerStackConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.behaviours import DynamicDiagonalOptions
from Emperor.linears.utils.layers import LinearLayerConfig, DynamicLinearLayerConfig
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)


class LinearsConfigs:
    @staticmethod
    def base_preset(
        batch_size=8,
        input_dim=12,
        output_dim=6,
        bias_flag=True,
        data_monitor=None,
        parameter_monitor=None,
    ) -> "ModelConfig":
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=output_dim,
            linear_layer_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=bias_flag,
                data_monitor=data_monitor,
                parameter_monitor=parameter_monitor,
            ),
        )

    @staticmethod
    def linear_layer_preset(
        input_dim=12,
        output_dim=6,
        bias_flag=True,
        data_monitor=None,
        parameter_monitor=None,
        *args,
        **kwargs,
    ):
        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            data_monitor=data_monitor,
            parameter_monitor=parameter_monitor,
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
        stack_depth: int = 2,
        stack_hidden_dim: int = 0,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
    ) -> "ModelConfig":
        stack_hidden_dim = stack_hidden_dim if stack_hidden_dim > 0 else input_dim

        # later use inspect to store all inputs of this function directly into the
        # linear
        # def function_b(input_dim, output_dim, *args, bias_flag=True, **kwargs):
        #     # Dynamically capture all arguments passed to function_b
        #     current_frame = inspect.currentframe()
        #     arg_values = inspect.getargvalues(current_frame)
        #
        #     # Extract arguments and their values
        #     all_args = arg_values.locals
        #
        #     # Forward all arguments to function_a
        #     function_a(**all_args)

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
                    num_layers=stack_depth,
                    activation=activation,
                    layer_norm_position=LayerNormPositionOptions.NONE,
                    residual_flag=residual_flag,
                    adaptive_computation_flag=False,
                    dropout_probability=dropout_probability,
                    override_config=DynamicLinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        generator_depth=generator_depth,
                        data_monitor=None,
                        parameter_monitor=None,
                    ),
                ),
            ),
            layer_stack_config=LayerStackConfig(
                model_type=LinearLayerOptions.BASE,
                input_dim=input_dim,
                hidden_dim=stack_hidden_dim,
                output_dim=output_dim,
                num_layers=stack_depth,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.NONE,
                residual_flag=residual_flag,
                adaptive_computation_flag=False,
                dropout_probability=dropout_probability,
            ),
        )


# class LinearsConfigs:
#     @staticmethod
#     def base_preset(
#         batch_size=8,
#         input_dim=12,
#         output_dim=6,
#         bias_flag=True,
#         data_monitor=None,
#         parameter_monitor=None,
#     ) -> "ModelConfig":
#         # Dynamically capture all arguments passed to base_preset
#         current_frame = inspect.currentframe()
#         arg_values = inspect.getargvalues(current_frame)
#
#         # Extract arguments as a dictionary
#         all_args = dict(arg_values.locals)
#
#         # Remove arguments that are not accepted by linear_layer_preset if needed (optional)
#         # This step is only required if extra arguments not suitable for `linear_layer_preset` exist
#         return ModelConfig(
#             batch_size=all_args["batch_size"],
#             input_dim=all_args["input_dim"],
#             output_dim=all_args["output_dim"],
#             linear_layer_config=LinearsConfigs.linear_layer_preset(**all_args),
#         )
#
#     @staticmethod
#     def linear_layer_preset(
#         input_dim=12,
#         output_dim=6,
#         bias_flag=True,
#         data_monitor=None,
#         parameter_monitor=None,
#         *args,
#         **kwargs,
#     ):
#         return LinearLayerConfig(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             bias_flag=bias_flag,
#             data_monitor=data_monitor,
#             parameter_monitor=parameter_monitor,
#         )
