"""Package-local adaptive input/output projection construction."""

from dataclasses import dataclass

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
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig

from models.experts.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.experts.linear_adaptive._adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.experts.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsStackOptions,
)


@dataclass(frozen=True, slots=True)
class AdaptiveBoundaryModelOptions:
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


@dataclass(frozen=True, slots=True)
class BoundaryModelConfigDependencies:
    stack_options: ExpertsStackOptions
    input_boundary_options: AdaptiveBoundaryModelOptions
    output_boundary_options: AdaptiveBoundaryModelOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions


class BoundaryModelConfigFactory:
    def __init__(self, dependencies: BoundaryModelConfigDependencies) -> None:
        self.stack_options = dependencies.stack_options
        self.input_boundary_options = dependencies.input_boundary_options
        self.output_boundary_options = dependencies.output_boundary_options
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
        )
        self.adaptive_generator_stack_config_factory = (
            AdaptiveGeneratorStackConfigFactory(
                dependencies.adaptive_generator_stack_options
            )
        )
        self.shared_adaptive_generator_stack_config = (
            self.adaptive_generator_stack_config_factory.build_shared_config()
        )

    def build_input_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            self.input_boundary_options,
            activation=self.stack_options.activation,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            self.output_boundary_options,
            activation=ActivationOptions.DISABLED,
        )

    def __build_boundary_layer_config(
        self,
        options: AdaptiveBoundaryModelOptions,
        *,
        activation: ActivationOptions,
    ) -> LayerConfig:
        return LayerConfig(
            activation=activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            layer_model_config=self.__build_boundary_layer_model_config(options),
        )

    def __build_boundary_layer_model_config(
        self,
        options: AdaptiveBoundaryModelOptions,
    ) -> AdaptiveLinearLayerConfig:
        augmentation = AdaptiveParameterAugmentationConfig(
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
            diagonal_config=build_diagonal_config(options.diagonal_option),
            mask_config=build_mask_config(
                row_mask_option=options.row_mask_option,
                mask_dimension_option=options.mask_dimension_option,
                mask_threshold=options.mask_threshold,
                mask_surrogate_scale=options.mask_surrogate_scale,
                mask_floor=options.mask_floor,
                mask_transition_width=options.mask_transition_width,
            ),
            model_config=self.shared_adaptive_generator_stack_config,
        )
        return AdaptiveLinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
            adaptive_augmentation_config=augmentation,
        )
