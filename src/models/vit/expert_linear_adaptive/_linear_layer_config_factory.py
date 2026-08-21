from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._adaptive_hidden_model_config_factory import (
    HiddenModelConfigDependencies,
    HiddenModelConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    TransformerEncoderOptions,
)

from ._residual import ResidualStackOptions, build_residual_config


@dataclass(frozen=True)
class LinearLayerConfigDependencies:
    encoder_options: TransformerEncoderOptions | None
    adaptive_augmentation_config: AdaptiveParameterAugmentationConfig | None = None


class LinearLayerConfigFactory:
    def __init__(self, dependencies: LinearLayerConfigDependencies) -> None:
        self.encoder_options = (
            config_defaults.vit_encoder_options(config)
            if dependencies.encoder_options is None
            else dependencies.encoder_options
        )
        self.hidden_dim = self.encoder_options.hidden_dim
        self.adaptive_augmentation_config = dependencies.adaptive_augmentation_config

    def build_backend_linear_layer_config(
        self,
        *,
        bias_flag: bool,
    ) -> LinearLayerConfig | AdaptiveLinearLayerConfig:
        if self.adaptive_augmentation_config is None:
            return self.build_plain_linear_layer_config(bias_flag=bias_flag)
        return AdaptiveLinearLayerConfig(
            bias_flag=bias_flag,
            adaptive_augmentation_config=self.adaptive_augmentation_config,
        )

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
        apply_output_postprocessing_flag: bool = True,
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
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
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
        apply_output_postprocessing_flag: bool = True,
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
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
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


@dataclass(frozen=True)
class AdaptiveAugmentationDependencies:
    hidden_dim: int
    output_dim: int
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None


class AdaptiveAugmentationConfigFactory:
    def __init__(self, dependencies: AdaptiveAugmentationDependencies) -> None:
        self.dependencies = dependencies
        config_module = config
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
            if dependencies.adaptive_generator_stack_options is not None
            else config_defaults.adaptive_generator_stack_options(config_module)
        )
        self.hidden_adaptive_weight_options = (
            dependencies.hidden_adaptive_weight_options
            if dependencies.hidden_adaptive_weight_options is not None
            else config_defaults.hidden_adaptive_weight_options(config_module)
        )
        self.hidden_adaptive_bias_options = (
            dependencies.hidden_adaptive_bias_options
            if dependencies.hidden_adaptive_bias_options is not None
            else config_defaults.hidden_adaptive_bias_options(config_module)
        )
        self.hidden_adaptive_diagonal_options = (
            dependencies.hidden_adaptive_diagonal_options
            if dependencies.hidden_adaptive_diagonal_options is not None
            else config_defaults.hidden_adaptive_diagonal_options(config_module)
        )
        self.hidden_adaptive_mask_options = (
            dependencies.hidden_adaptive_mask_options
            if dependencies.hidden_adaptive_mask_options is not None
            else config_defaults.hidden_adaptive_mask_options(config_module)
        )

    def build_adaptive_augmentation_config(
        self,
    ) -> AdaptiveParameterAugmentationConfig:
        dependencies = self.dependencies
        factory = HiddenModelConfigFactory(
            HiddenModelConfigDependencies(
                hidden_dim=dependencies.hidden_dim,
                stack_options=None,
                submodule_stack_options=None,
                layer_controller_options=None,
                dynamic_memory_options=None,
                recurrent_controller_options=None,
                hidden_adaptive_weight_options=(self.hidden_adaptive_weight_options),
                hidden_adaptive_bias_options=(self.hidden_adaptive_bias_options),
                hidden_adaptive_diagonal_options=(
                    self.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                adaptive_generator_stack_options=(
                    self.adaptive_generator_stack_options
                ),
                output_dim=dependencies.output_dim,
            )
        )
        return factory.adaptive_augmentation_config
