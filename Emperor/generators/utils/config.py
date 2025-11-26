from Emperor.base.layer import LayerStackConfig
from Emperor.linears.options import LinearLayerOptions
from Emperor.generators.utils.routers import RouterConfig
from Emperor.linears.utils.layers import DynamicLinearLayerConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class GeneratorConfigs:
    @staticmethod
    def router_preset(
        batch_size: int = 2,
        input_dim: int = 12,
        hidden_dim: int = 16,
        num_experts: int = 6,
        bias_flag: bool = True,
        noisy_topk_flag: bool = False,
        residual_flag: bool = True,
        activation: ActivationOptions = ActivationOptions.RELU,
        model_type: LinearLayerOptions = LinearLayerOptions.BASE,
        bias_option: DynamicBiasOptions = DynamicBiasOptions.DISABLED,
        memory_option: LinearMemoryOptions = LinearMemoryOptions.DISABLED,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
        diagonal_option: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED,
        memory_size_option: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED,
        memory_position_option: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE,
    ) -> "ModelConfig":
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
            router_model_config=RouterConfig(
                num_experts=num_experts,
                noisy_topk_flag=noisy_topk_flag,
            ),
            layer_stack_config=LayerStackConfig(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=num_experts,
                num_layers=2,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.NONE,
                residual_flag=residual_flag,
                adaptive_computation_flag=False,
            ),
            linear_layer_config=DynamicLinearLayerConfig(
                input_dim=input_dim,
                output_dim=num_experts,
                bias_flag=bias_flag,
                data_monitor=None,
                parameter_monitor=None,
                generator_depth=generator_depth,
                diagonal_option=diagonal_option,
                bias_option=bias_option,
                memory_option=memory_option,
                memory_size_option=memory_size_option,
                memory_position_option=memory_position_option,
            ),
        )
