from dataclasses import dataclass
from typing import TYPE_CHECKING

import models.linears.linear_adaptive.config as config
from models.linears._builder_options import MainLayerStackOptions
from emperor.augmentations.adaptive_parameters.core.bias import DynamicBiasConfig
from emperor.augmentations.adaptive_parameters.core.mask import AxisMaskConfig
from emperor.augmentations.adaptive_parameters.core.weight import DynamicWeightConfig
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from models.linears.linear_adaptive._adaptive_generator_stack_config_factory import (
    AdaptiveGeneratorStackConfigFactory,
)
from models.linears.linear_adaptive._builder_options import (
    AdaptiveGeneratorStackOptions,
)
from models.adaptive_parameter_config_factory import (
    build_bias_config,
    build_diagonal_config,
    build_mask_config,
    build_weight_config,
)

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class BoundaryModelConfigDependencies:
    stack_options: MainLayerStackOptions | None
    input_boundary_options: AdaptiveBoundaryModelOptions | None
    output_boundary_options: AdaptiveBoundaryModelOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None


class BoundaryModelConfigFactory:
    def __init__(self, dependencies: BoundaryModelConfigDependencies) -> None:
        stack_options = self.__default_stack_options(dependencies.stack_options)
        input_boundary_options = self.__default_input_boundary_options(
            dependencies.input_boundary_options
        )
        output_boundary_options = self.__default_output_boundary_options(
            dependencies.output_boundary_options
        )
        adaptive_generator_stack_options = (
            self.__default_adaptive_generator_stack_options(
                dependencies.adaptive_generator_stack_options
            )
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
            self.__build_shared_adaptive_generator_stack_config()
        )
        self.shared_adaptive_generator_stack_config = (
            shared_adaptive_generator_stack_config
        )

    def __default_stack_options(
        self,
        stack_options: MainLayerStackOptions | None,
    ) -> MainLayerStackOptions:
        if stack_options is not None:
            return stack_options
        return MainLayerStackOptions(
            bias_flag=config.STACK_BIAS_FLAG,
            layer_norm_position=config.STACK_LAYER_NORM_POSITION,
            num_layers=config.STACK_NUM_LAYERS,
            activation=config.STACK_ACTIVATION,
            residual_connection_option=config.STACK_RESIDUAL_CONNECTION_OPTION,
            dropout_probability=config.STACK_DROPOUT_PROBABILITY,
            last_layer_bias_option=config.STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_pipeline_flag=config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        )

    def __default_adaptive_generator_stack_options(
        self,
        adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None,
    ) -> AdaptiveGeneratorStackOptions:
        if adaptive_generator_stack_options is not None:
            return adaptive_generator_stack_options
        return AdaptiveGeneratorStackOptions(
            hidden_dim=config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM,
            layer_norm_position=(config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION),
            num_layers=config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS,
            activation=config.ADAPTIVE_GENERATOR_STACK_ACTIVATION,
            residual_connection_option=(
                config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
            ),
            dropout_probability=(config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY),
            last_layer_bias_option=(
                config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
            ),
            apply_output_pipeline_flag=(
                config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
            ),
            bias_flag=config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG,
        )

    def __default_input_boundary_options(
        self,
        boundary_options: AdaptiveBoundaryModelOptions | None,
    ) -> AdaptiveBoundaryModelOptions:
        if boundary_options is not None:
            return boundary_options
        return AdaptiveBoundaryModelOptions(
            weight_option=config.INPUT_LAYER_WEIGHT_OPTION,
            weight_generator_depth=config.INPUT_LAYER_WEIGHT_GENERATOR_DEPTH,
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
            weight_generator_depth=config.OUTPUT_LAYER_WEIGHT_GENERATOR_DEPTH,
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
            residual_connection_option=ResidualConnectionOptions.DISABLED,
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
            model_config=self.shared_adaptive_generator_stack_config,
        )
        return AdaptiveLinearLayerConfig(
            bias_flag=self.stack_options.bias_flag,
            adaptive_augmentation_config=adaptive_augmentation_config,
        )

    def __build_shared_adaptive_generator_stack_config(self) -> LayerStackConfig:
        return self.adaptive_generator_stack_config_factory.build_shared_config()
