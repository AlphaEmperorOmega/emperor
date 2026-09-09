from emperor.layers import LayerConfig, LayerStackConfig
from emperor.linears import LinearLayerConfig
from models.gpt.linear_adaptive.runtime_options import SubmoduleStackOptions

from ._residual import build_residual_config


def build_controller_stack_config(
    options: SubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    """Build the package-local stack used by controller models."""

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
            ),
            dropout_probability=options.dropout_probability,
            halting_config=None,
            gate_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
        ),
    )
