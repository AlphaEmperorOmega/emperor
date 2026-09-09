from emperor.config import ConfigBase
from emperor.layers import LayerConfig, LayerStackConfig
from emperor.linears import LinearLayerConfig
from models.bert.expert_linear.runtime_options import (
    ExpertsSubmoduleStackOptions,
    SubmoduleStackOptions,
)

from ._residual import build_residual_config


def build_controller_stack_config(
    options: SubmoduleStackOptions | ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    layer_model_config: ConfigBase | None = None,
) -> LayerStackConfig:
    """Build a regular or expert controller stack for this Model Package."""

    return LayerStackConfig(
        hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_postprocessing_flag=options.apply_output_postprocessing_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            normalization=options.normalization,
            residual_config=build_residual_config(
                options.residual_connection_option,
                options.residual_model_flag,
                options.residual_stack_options,
                residual_block_size=options.residual_block_size,
                residual_rms_norm_epsilon=options.residual_rms_norm_epsilon,
            ),
            dropout_probability=options.dropout_probability,
            halting_config=None,
            gate_config=None,
            memory_config=None,
            layer_model_config=(
                LinearLayerConfig(bias_flag=options.bias_flag)
                if layer_model_config is None
                else layer_model_config
            ),
        ),
    )
