from dataclasses import dataclass

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)


@dataclass(frozen=True)
class SubmoduleStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_pipeline_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: ResidualConnectionOptions | None
    dropout_probability: float | None
    bias_flag: bool | None


@dataclass(frozen=True)
class SubmoduleStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    bias_flag: bool


def resolve_controller_stack_options(
    source: SubmoduleStackSource,
    defaults: SubmoduleStackOptions,
) -> SubmoduleStackOptions:
    if not source.independent_flag:
        return defaults
    hidden_dim = defaults.hidden_dim if source.hidden_dim is None else source.hidden_dim
    num_layers = defaults.num_layers if source.num_layers is None else source.num_layers
    last_layer_bias_option = (
        defaults.last_layer_bias_option
        if source.last_layer_bias_option is None
        else source.last_layer_bias_option
    )
    apply_output_pipeline_flag = (
        defaults.apply_output_pipeline_flag
        if source.apply_output_pipeline_flag is None
        else source.apply_output_pipeline_flag
    )
    activation = defaults.activation if source.activation is None else source.activation
    layer_norm_position = (
        defaults.layer_norm_position
        if source.layer_norm_position is None
        else source.layer_norm_position
    )
    residual_connection_option = (
        defaults.residual_connection_option
        if source.residual_connection_option is None
        else source.residual_connection_option
    )
    dropout_probability = (
        defaults.dropout_probability
        if source.dropout_probability is None
        else source.dropout_probability
    )
    bias_flag = defaults.bias_flag if source.bias_flag is None else source.bias_flag
    return SubmoduleStackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        activation=activation,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        dropout_probability=dropout_probability,
        bias_flag=bias_flag,
    )
