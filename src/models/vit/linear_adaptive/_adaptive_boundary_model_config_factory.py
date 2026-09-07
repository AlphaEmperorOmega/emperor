from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import models.vit.linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GroupingConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LayerConfig,
    LayerNormPositionOptions,
)
from models.vit.linear_adaptive import _config_defaults as config_defaults
from models.vit.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.vit.linear_adaptive._adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)
from models.vit.linear_adaptive._generation import (
    BiasGenerationOptions,
    WeightGenerationOptions,
    mixture_generation_fields,
    weight_generation_fields,
)
from models.vit.linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    MainLayerStackOptions,
)

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig


@dataclass(frozen=True)
class AdaptiveBoundaryModelOptions:
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    weight_generation: WeightGenerationOptions = field(
        default_factory=WeightGenerationOptions, kw_only=True
    )
    bias_generation: BiasGenerationOptions = field(
        default_factory=BiasGenerationOptions, kw_only=True
    )
    weight_option: type[DynamicWeightConfig] | None
    generator_depth: DynamicDepthOptions
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


@dataclass(frozen=True)
class BoundaryModelConfigDependencies:
    stack_options: MainLayerStackOptions | None
    input_boundary_options: AdaptiveBoundaryModelOptions | None
    output_boundary_options: AdaptiveBoundaryModelOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None


class BoundaryModelConfigFactory:
    def __init__(self, dependencies: BoundaryModelConfigDependencies) -> None:
        stack_options = (
            config_defaults.main_layer_stack_options(config)
            if dependencies.stack_options is None
            else dependencies.stack_options
        )
        input_boundary_options = self.__default_input_boundary_options(
            dependencies.input_boundary_options
        )
        output_boundary_options = self.__default_output_boundary_options(
            dependencies.output_boundary_options
        )
        adaptive_generator_stack_options = (
            config_defaults.adaptive_generator_stack_options(config)
            if dependencies.adaptive_generator_stack_options is None
            else dependencies.adaptive_generator_stack_options
        )
        adaptive_generator_stack_config_factory = AdaptiveGeneratorStackConfigFactory(
            adaptive_generator_stack_options
        )

        self.stack_options = stack_options
        self.input_boundary_options = input_boundary_options
        self.output_boundary_options = output_boundary_options
        self.adaptive_generator_stack_options = adaptive_generator_stack_options
        self.adaptive_generator_stack_config_factory = (
            adaptive_generator_stack_config_factory
        )
        shared_adaptive_generator_stack_config = (
            self.adaptive_generator_stack_config_factory.build_shared_config()
        )
        self.shared_adaptive_generator_stack_config = (
            shared_adaptive_generator_stack_config
        )

    def __default_input_boundary_options(
        self,
        boundary_options: AdaptiveBoundaryModelOptions | None,
    ) -> AdaptiveBoundaryModelOptions:
        if boundary_options is not None:
            return boundary_options
        return AdaptiveBoundaryModelOptions(
            weight_option=config.INPUT_LAYER_WEIGHT_OPTION,
            generator_depth=config.INPUT_LAYER_GENERATOR_DEPTH,
            weight_decay_schedule=config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
            weight_decay_rate=config.INPUT_LAYER_WEIGHT_DECAY_RATE,
            weight_decay_warmup_batches=(
                config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
            ),
            weight_normalization_option=(
                config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
            ),
            weight_normalization_position_option=(
                config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
            ),
            weight_bank_expansion_factor=(
                config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
            ),
            bias_option=config.INPUT_LAYER_BIAS_OPTION,
            bias_decay_schedule=config.INPUT_LAYER_BIAS_DECAY_SCHEDULE,
            bias_decay_rate=config.INPUT_LAYER_BIAS_DECAY_RATE,
            bias_decay_warmup_batches=(config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES),
            bias_bank_expansion_factor=(config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR),
            diagonal_option=config.INPUT_LAYER_DIAGONAL_OPTION,
            row_mask_option=config.INPUT_LAYER_ROW_MASK_OPTION,
            mask_dimension_option=config.INPUT_LAYER_MASK_DIMENSION_OPTION,
            mask_threshold=config.INPUT_LAYER_MASK_THRESHOLD,
            mask_surrogate_scale=config.INPUT_LAYER_MASK_SURROGATE_SCALE,
            mask_floor=config.INPUT_LAYER_MASK_FLOOR,
            mask_transition_width=config.INPUT_LAYER_MASK_TRANSITION_WIDTH,
        )

    def __default_output_boundary_options(
        self,
        boundary_options: AdaptiveBoundaryModelOptions | None,
    ) -> AdaptiveBoundaryModelOptions:
        if boundary_options is not None:
            return boundary_options
        return AdaptiveBoundaryModelOptions(
            weight_option=config.OUTPUT_LAYER_WEIGHT_OPTION,
            generator_depth=config.OUTPUT_LAYER_GENERATOR_DEPTH,
            weight_decay_schedule=config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE,
            weight_decay_rate=config.OUTPUT_LAYER_WEIGHT_DECAY_RATE,
            weight_decay_warmup_batches=(
                config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
            ),
            weight_normalization_option=(
                config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
            ),
            weight_normalization_position_option=(
                config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
            ),
            weight_bank_expansion_factor=(
                config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
            ),
            bias_option=config.OUTPUT_LAYER_BIAS_OPTION,
            bias_decay_schedule=config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE,
            bias_decay_rate=config.OUTPUT_LAYER_BIAS_DECAY_RATE,
            bias_decay_warmup_batches=(config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES),
            bias_bank_expansion_factor=(config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR),
            diagonal_option=config.OUTPUT_LAYER_DIAGONAL_OPTION,
            row_mask_option=config.OUTPUT_LAYER_ROW_MASK_OPTION,
            mask_dimension_option=config.OUTPUT_LAYER_MASK_DIMENSION_OPTION,
            mask_threshold=config.OUTPUT_LAYER_MASK_THRESHOLD,
            mask_surrogate_scale=config.OUTPUT_LAYER_MASK_SURROGATE_SCALE,
            mask_floor=config.OUTPUT_LAYER_MASK_FLOOR,
            mask_transition_width=config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH,
        )

    def build_input_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            options=self.input_boundary_options,
            activation=self.stack_options.activation,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
        )

    def build_output_model_config(self) -> LayerConfig:
        return self.__build_boundary_layer_config(
            options=self.output_boundary_options,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
        )

    def __build_boundary_layer_config(
        self,
        options: AdaptiveBoundaryModelOptions,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
        gate_config: GateConfig | None,
        halting_config: "HaltingConfig | None",
    ) -> LayerConfig:
        layer_model_config = self.__build_boundary_layer_model_config(options)
        return LayerConfig(
            activation=activation,
            layer_norm_position=layer_norm_position,
            residual_config=None,
            dropout_probability=dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=layer_model_config,
        )

    def __build_boundary_layer_model_config(
        self,
        options: AdaptiveBoundaryModelOptions,
    ) -> AdaptiveLinearLayerConfig:
        adaptive_augmentation_config = AdaptiveParameterAugmentationConfig(
            grouping_config=options.grouping_config,
            weight_config=build_weight_config(
                generation_fields=weight_generation_fields(
                    options.weight_generation,
                    self.adaptive_generator_stack_config_factory,
                ),
                weight_option=options.weight_option,
                generator_depth=options.generator_depth,
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
                generation_fields=mixture_generation_fields(
                    options.bias_generation,
                    self.adaptive_generator_stack_config_factory,
                ),
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
            model_config=self.shared_adaptive_generator_stack_config,
        )
        return AdaptiveLinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )
