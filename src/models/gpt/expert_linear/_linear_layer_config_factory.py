from dataclasses import dataclass

import models.gpt.expert_linear.config as config
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from models.gpt.expert_linear.runtime_options import TransformerDecoderOptions


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    decoder_options: TransformerDecoderOptions | None


class LinearLayerConfigFactory:
    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        self.decoder_options = self.__default_decoder_options(
            dependencies.decoder_options
        )
        self.hidden_dim = self.decoder_options.hidden_dim

    def __default_decoder_options(
        self,
        decoder_options: TransformerDecoderOptions | None,
    ) -> TransformerDecoderOptions:
        if decoder_options is not None:
            return decoder_options
        return TransformerDecoderOptions(
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            layer_norm_position=config.LAYER_NORM_POSITION,
        )

    def build_backend_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig:
        return self.build_plain_linear_layer_config(bias_flag=bias_flag)

    def build_plain_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig:
        return LinearLayerConfig(bias_flag=bias_flag)

    def build_backend_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        return self.build_linear_stack_config(
            layer_model_config=self.build_backend_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
        )

    def build_plain_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        return self.build_linear_stack_config(
            layer_model_config=self.build_plain_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
        )

    def build_linear_stack_config(
        self,
        *,
        layer_model_config,
        num_layers: int,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: ActivationOptions | None = None,
        residual_connection_option: ResidualConnectionOptions | None = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        layer_config = LayerConfig(
            activation=(
                self.decoder_options.activation if activation is None else activation
            ),
            layer_norm_position=layer_norm_position,
            residual_config=None
            if residual_connection_option is None
            else ResidualConfig(option=residual_connection_option),
            dropout_probability=dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=layer_model_config,
        )
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim if hidden_dim is None else hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=layer_config,
        )
