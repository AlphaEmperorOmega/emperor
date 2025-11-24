from typing import TYPE_CHECKING

from Emperor.generators.utils.routers import RouterConfig
from Emperor.linears.utils.layers import DynamicLinearLayerConfig

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class GeneratorConfigs:
    @staticmethod
    def router_preset(
        batch_size=2,
        input_dim=12,
        hidden_dim=16,
        num_experts=6,
        bias_flag=True,
        data_monitor=None,
        parameter_monitor=None,
    ) -> "ModelConfig":
        return ModelConfig(
            batch_size=batch_size,
            input_dim=input_dim,
            output_dim=num_experts,
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
            ),
            router_model_config=RouterConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                residual_flag=ROUTER_RESIDUAL_FLAG,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION,
                num_layers=ROUTER_NUM_LAYERS,
                diagonal_model_type_flag=ROUTER_DIAGONAL_LINEAR_MODEL_FLAG,
            ),
        )
