from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.core.weight import DynamicWeightConfig
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.layer.config import LayerConfig
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig


@dataclass(frozen=True)
class AdaptiveBoundaryProjectionOptions:
    weight_option: type[DynamicWeightConfig] | None
    weight_generator_depth: DynamicDepthOptions
    weight_decay_schedule: WeightDecayScheduleOptions
    weight_decay_rate: float
    weight_decay_warmup_batches: int
    weight_normalization_option: WeightNormalizationOptions
    weight_normalization_position_option: WeightNormalizationPositionOptions
    weight_bank_expansion_factor: BankExpansionFactorOptions
    bias_option: type[DynamicBiasConfig] | None
    bias_decay_schedule: WeightDecayScheduleOptions
    bias_decay_rate: float
    bias_decay_warmup_batches: int
    bias_bank_expansion_factor: BankExpansionFactorOptions
    diagonal_option: type[DynamicDiagonalConfig] | None
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float


BoundaryLayerOptions = AdaptiveBoundaryProjectionOptions


class BoundaryConfigFactory:
    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def build_input_model_config(self) -> LayerConfig:
        return self._build_boundary_layer_config(
            options=self.builder.input_boundary_options,
            activation=self.builder.stack_activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self._build_boundary_layer_config(
            options=self.builder.output_boundary_options,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
        )

    def _build_boundary_layer_config(
        self,
        options: AdaptiveBoundaryProjectionOptions,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        gate_config: GateConfig | None,
        halting_config: "HaltingConfig | None",
    ) -> LayerConfig:
        return LayerConfig(
            activation=activation,
            layer_norm_position=layer_norm_position,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=self._build_boundary_layer_model_config(options),
        )

    def _build_boundary_layer_model_config(
        self,
        options: AdaptiveBoundaryProjectionOptions,
    ) -> AdaptiveLinearLayerConfig:
        builder = self.builder
        return AdaptiveLinearLayerConfig(
            bias_flag=builder.bias_flag,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=build_weight_config(
                    weight_option=options.weight_option,
                    generator_depth=options.weight_generator_depth,
                    decay_schedule=options.weight_decay_schedule,
                    decay_rate=options.weight_decay_rate,
                    decay_warmup_batches=options.weight_decay_warmup_batches,
                    normalization_option=options.weight_normalization_option,
                    normalization_position_option=(
                        options.weight_normalization_position_option
                    ),
                    bank_expansion_factor=options.weight_bank_expansion_factor,
                ),
                bias_config=build_bias_config(
                    bias_option=options.bias_option,
                    decay_schedule=options.bias_decay_schedule,
                    decay_rate=options.bias_decay_rate,
                    decay_warmup_batches=options.bias_decay_warmup_batches,
                    bank_expansion_factor=options.bias_bank_expansion_factor,
                ),
                diagonal_config=build_diagonal_config(
                    options.diagonal_option,
                ),
                mask_config=build_mask_config(
                    row_mask_option=options.row_mask_option,
                    mask_dimension_option=options.mask_dimension_option,
                    mask_threshold=options.mask_threshold,
                    mask_surrogate_scale=options.mask_surrogate_scale,
                    mask_floor=options.mask_floor,
                    mask_transition_width=options.mask_transition_width,
                ),
                model_config=builder._build_model_config(),
            ),
        )
