from dataclasses import dataclass

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)


@dataclass(frozen=True)
class AdaptiveGeneratorStackSource:
    independent_flag: bool
    hidden_dim: int | None
    layer_norm_position: LayerNormPositionOptions | None
    num_layers: int | None
    activation: ActivationOptions | None
    residual_connection_option: ResidualConnectionOptions | None
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_pipeline_flag: bool | None
    bias_flag: bool | None


@dataclass(frozen=True)
class AdaptiveGeneratorStackOptions:
    hidden_dim: int
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    bias_flag: bool
