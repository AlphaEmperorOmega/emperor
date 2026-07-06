from emperor.base.layer.config import LayerConfig, LayerStackConfig

from models.bert._tokenwise import BertTokenwiseLinearLayerConfig
from models.experts._builder_options import ExpertsSubmoduleStackOptions


def build_linear_controller_stack(
    options: ExpertsSubmoduleStackOptions,
    *,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=BertTokenwiseLinearLayerConfig(
                bias_flag=options.bias_flag,
            ),
        ),
    )


__all__ = ["build_linear_controller_stack"]
