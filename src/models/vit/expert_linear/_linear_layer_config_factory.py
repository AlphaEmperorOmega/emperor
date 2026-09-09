from dataclasses import dataclass

import models.vit.expert_linear.config as config
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from models.vit.expert_linear._config_defaults import vit_encoder_options
from models.vit.expert_linear.runtime_options import TransformerEncoderOptions

from ._residual import ResidualStackOptions, build_residual_config


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    encoder_options: TransformerEncoderOptions | None


class LinearLayerConfigFactory:
    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        self.encoder_options = (
            vit_encoder_options(config)
            if dependencies.encoder_options is None
            else dependencies.encoder_options
        )
        self.hidden_dim = self.encoder_options.hidden_dim

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
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        return self.build_linear_stack_config(
            layer_model_config=self.build_backend_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        )

    def build_plain_linear_stack_config(
        self,
        *,
        num_layers: int,
        bias_flag: bool,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        output_dim: int | None = None,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        return self.build_linear_stack_config(
            layer_model_config=self.build_plain_linear_layer_config(
                bias_flag=bias_flag,
            ),
            num_layers=num_layers,
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            dropout_probability=dropout_probability,
            input_dim=input_dim,
            output_dim=output_dim,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        )

    def build_linear_stack_config(
        self,
        *,
        layer_model_config,
        num_layers: int,
        layer_norm_position: LayerNormPositionOptions,
        normalization: NormalizationOptions = NormalizationOptions.RMS_NORM,
        dropout_probability: float,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        activation: ActivationOptions | None = None,
        residual_connection_option: type[ResidualConfig] | None = None,
        residual_model_flag: bool = False,
        residual_stack_options: ResidualStackOptions | None = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag: bool = True,
    ) -> LayerStackConfig:
        layer_config = LayerConfig(
            activation=(
                self.encoder_options.activation if activation is None else activation
            ),
            layer_norm_position=layer_norm_position,
            normalization=normalization,
            residual_config=build_residual_config(
                residual_connection_option,
                residual_model_flag,
                residual_stack_options,
            ),
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
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            layer_config=layer_config,
        )
