from dataclasses import dataclass, field
from typing import cast

import models.neuron.expert_linear_adaptive.config as config
from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from models.neuron.expert_linear_adaptive._hidden._projection_config_factory import (
    AdaptiveBoundaryModelOptions,
)
from models.neuron.expert_linear_adaptive._hidden.runtime_options import (
    AdaptiveGeneratorStackOptions,
    AdaptiveGeneratorStackSource,
    ExpertsAdaptiveGeneratorStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
)


@dataclass(frozen=True, kw_only=True)
class AdaptiveDefaultValues:
    weight_option_flag: bool | None = None
    generator_depth: DynamicDepthOptions = config.GENERATOR_DEPTH
    diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION
    diagonal_option_flag: bool | None = None
    bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION
    bias_option_flag: bool | None = None
    weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION
    weight_generator_stack_independent_flag: bool = (
        config.WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    weight_generator_stack_hidden_dim: int | None = (
        config.WEIGHT_GENERATOR_STACK_HIDDEN_DIM
    )
    weight_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    weight_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.WEIGHT_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    weight_generator_stack_num_layers: int | None = (
        config.WEIGHT_GENERATOR_STACK_NUM_LAYERS
    )
    weight_generator_stack_activation: ActivationOptions | None = (
        config.WEIGHT_GENERATOR_STACK_ACTIVATION
    )
    weight_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    weight_generator_stack_residual_model_flag: bool = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    weight_generator_stack_residual_block_size: int | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    weight_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    weight_generator_stack_dropout_probability: float | None = (
        config.WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    weight_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    weight_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    weight_generator_stack_bias_flag: bool | None = (
        config.WEIGHT_GENERATOR_STACK_BIAS_FLAG
    )
    weight_normalization_option: WeightNormalizationOptions = (
        config.WEIGHT_NORMALIZATION_OPTION
    )
    weight_normalization_position_option: WeightNormalizationPositionOptions = (
        config.WEIGHT_NORMALIZATION_POSITION_OPTION
    )
    weight_decay_schedule: WeightDecayScheduleOptions = config.WEIGHT_DECAY_SCHEDULE
    weight_decay_rate: float = config.WEIGHT_DECAY_RATE
    weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES
    weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.WEIGHT_BANK_EXPANSION_FACTOR
    )
    bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE
    bias_decay_rate: float = config.BIAS_DECAY_RATE
    bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES
    bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.BIAS_BANK_EXPANSION_FACTOR
    )
    bias_generator_stack_independent_flag: bool = (
        config.BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    bias_generator_stack_hidden_dim: int | None = config.BIAS_GENERATOR_STACK_HIDDEN_DIM
    bias_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    bias_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.BIAS_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    bias_generator_stack_num_layers: int | None = config.BIAS_GENERATOR_STACK_NUM_LAYERS
    bias_generator_stack_activation: ActivationOptions | None = (
        config.BIAS_GENERATOR_STACK_ACTIVATION
    )
    bias_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    bias_generator_stack_residual_model_flag: bool = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    bias_generator_stack_residual_block_size: int | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    bias_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    bias_generator_stack_dropout_probability: float | None = (
        config.BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    bias_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    bias_generator_stack_bias_flag: bool | None = config.BIAS_GENERATOR_STACK_BIAS_FLAG
    diagonal_generator_stack_independent_flag: bool = (
        config.DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    diagonal_generator_stack_hidden_dim: int | None = (
        config.DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
    )
    diagonal_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    diagonal_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.DIAGONAL_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    diagonal_generator_stack_num_layers: int | None = (
        config.DIAGONAL_GENERATOR_STACK_NUM_LAYERS
    )
    diagonal_generator_stack_activation: ActivationOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_ACTIVATION
    )
    diagonal_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    diagonal_generator_stack_residual_model_flag: bool = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    diagonal_generator_stack_residual_block_size: int | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    diagonal_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    diagonal_generator_stack_dropout_probability: float | None = (
        config.DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    diagonal_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    diagonal_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    diagonal_generator_stack_bias_flag: bool | None = (
        config.DIAGONAL_GENERATOR_STACK_BIAS_FLAG
    )
    row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION
    mask_option_flag: bool | None = None
    mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION
    mask_threshold: float = config.MASK_THRESHOLD
    mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE
    mask_floor: float = config.MASK_FLOOR
    mask_transition_width: float = config.MASK_TRANSITION_WIDTH
    mask_generator_stack_independent_flag: bool = (
        config.MASK_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    mask_generator_stack_hidden_dim: int | None = config.MASK_GENERATOR_STACK_HIDDEN_DIM
    mask_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.MASK_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    mask_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.MASK_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    mask_generator_stack_num_layers: int | None = config.MASK_GENERATOR_STACK_NUM_LAYERS
    mask_generator_stack_activation: ActivationOptions | None = (
        config.MASK_GENERATOR_STACK_ACTIVATION
    )
    mask_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    mask_generator_stack_residual_model_flag: bool = (
        config.MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    mask_generator_stack_residual_block_size: int | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    mask_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    mask_generator_stack_dropout_probability: float | None = (
        config.MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    mask_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    mask_generator_stack_bias_flag: bool | None = config.MASK_GENERATOR_STACK_BIAS_FLAG
    adaptive_generator_stack_num_layers: int = (
        config.ADAPTIVE_GENERATOR_STACK_NUM_LAYERS
    )
    adaptive_generator_stack_hidden_dim: int = (
        config.ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM
    )
    adaptive_generator_stack_activation: ActivationOptions = (
        config.ADAPTIVE_GENERATOR_STACK_ACTIVATION
    )
    adaptive_generator_stack_residual_connection_option: type[ResidualConfig] | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    )
    adaptive_generator_stack_residual_model_flag: bool = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    adaptive_generator_stack_residual_block_size: int | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    adaptive_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    adaptive_generator_stack_dropout_probability: float = (
        config.ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = (
        config.ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    adaptive_generator_stack_normalization: NormalizationOptions = field(
        default=(config.ADAPTIVE_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = (
        config.ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    adaptive_generator_stack_apply_output_postprocessing_flag: bool = (
        config.ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    adaptive_generator_stack_bias_flag: bool = config.ADAPTIVE_GENERATOR_STACK_BIAS_FLAG
    input_layer_weight_option: type[DynamicWeightConfig] | None = (
        config.INPUT_LAYER_WEIGHT_OPTION
    )
    input_layer_generator_depth: DynamicDepthOptions = (
        config.INPUT_LAYER_GENERATOR_DEPTH
    )
    input_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE
    )
    input_layer_weight_decay_rate: float = config.INPUT_LAYER_WEIGHT_DECAY_RATE
    input_layer_weight_decay_warmup_batches: int = (
        config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
    )
    input_layer_weight_normalization_option: WeightNormalizationOptions = (
        config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
    )
    input_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
    input_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    input_layer_bias_option: type[DynamicBiasConfig] | None = (
        config.INPUT_LAYER_BIAS_OPTION
    )
    input_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.INPUT_LAYER_BIAS_DECAY_SCHEDULE
    )
    input_layer_bias_decay_rate: float = config.INPUT_LAYER_BIAS_DECAY_RATE
    input_layer_bias_decay_warmup_batches: int = (
        config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
    )
    input_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
    )
    input_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.INPUT_LAYER_DIAGONAL_OPTION
    )
    input_layer_row_mask_option: type[AxisMaskConfig] | None = (
        config.INPUT_LAYER_ROW_MASK_OPTION
    )
    input_layer_mask_dimension_option: MaskDimensionOptions = (
        config.INPUT_LAYER_MASK_DIMENSION_OPTION
    )
    input_layer_mask_threshold: float = config.INPUT_LAYER_MASK_THRESHOLD
    input_layer_mask_surrogate_scale: float = config.INPUT_LAYER_MASK_SURROGATE_SCALE
    input_layer_mask_floor: float = config.INPUT_LAYER_MASK_FLOOR
    input_layer_mask_transition_width: float = config.INPUT_LAYER_MASK_TRANSITION_WIDTH
    output_layer_weight_option: type[DynamicWeightConfig] | None = (
        config.OUTPUT_LAYER_WEIGHT_OPTION
    )
    output_layer_generator_depth: DynamicDepthOptions = (
        config.OUTPUT_LAYER_GENERATOR_DEPTH
    )
    output_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE
    )
    output_layer_weight_decay_rate: float = config.OUTPUT_LAYER_WEIGHT_DECAY_RATE
    output_layer_weight_decay_warmup_batches: int = (
        config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
    )
    output_layer_weight_normalization_option: WeightNormalizationOptions = (
        config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
    )
    output_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
    output_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    output_layer_bias_option: type[DynamicBiasConfig] | None = (
        config.OUTPUT_LAYER_BIAS_OPTION
    )
    output_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE
    )
    output_layer_bias_decay_rate: float = config.OUTPUT_LAYER_BIAS_DECAY_RATE
    output_layer_bias_decay_warmup_batches: int = (
        config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
    )
    output_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
    )
    output_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.OUTPUT_LAYER_DIAGONAL_OPTION
    )
    output_layer_row_mask_option: type[AxisMaskConfig] | None = (
        config.OUTPUT_LAYER_ROW_MASK_OPTION
    )
    output_layer_mask_dimension_option: MaskDimensionOptions = (
        config.OUTPUT_LAYER_MASK_DIMENSION_OPTION
    )
    output_layer_mask_threshold: float = config.OUTPUT_LAYER_MASK_THRESHOLD
    output_layer_mask_surrogate_scale: float = config.OUTPUT_LAYER_MASK_SURROGATE_SCALE
    output_layer_mask_floor: float = config.OUTPUT_LAYER_MASK_FLOOR
    output_layer_mask_transition_width: float = (
        config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH
    )
    router_weight_option_flag: bool | None = None
    router_weight_option: type[DynamicWeightConfig] | None = config.ROUTER_WEIGHT_OPTION
    router_generator_depth: DynamicDepthOptions = config.ROUTER_GENERATOR_DEPTH
    router_weight_normalization_option: WeightNormalizationOptions = (
        config.ROUTER_WEIGHT_NORMALIZATION_OPTION
    )
    router_weight_normalization_position_option: WeightNormalizationPositionOptions = (
        config.ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION
    )
    router_weight_decay_schedule: WeightDecayScheduleOptions = (
        config.ROUTER_WEIGHT_DECAY_SCHEDULE
    )
    router_weight_decay_rate: float = config.ROUTER_WEIGHT_DECAY_RATE
    router_weight_decay_warmup_batches: int = config.ROUTER_WEIGHT_DECAY_WARMUP_BATCHES
    router_weight_bank_expansion_factor: BankExpansionFactorOptions = (
        config.ROUTER_WEIGHT_BANK_EXPANSION_FACTOR
    )
    router_weight_generator_stack_independent_flag: bool = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_weight_generator_stack_hidden_dim: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM
    )
    router_weight_generator_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
    router_weight_generator_stack_normalization: NormalizationOptions | None = field(
        default=config.ROUTER_WEIGHT_GENERATOR_STACK_NORMALIZATION, kw_only=True
    )
    router_weight_generator_stack_num_layers: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS
    )
    router_weight_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION
    )
    router_weight_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_weight_generator_stack_residual_model_flag: bool = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_weight_generator_stack_residual_block_size: int | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_weight_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_weight_generator_stack_dropout_probability: float | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_weight_generator_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    router_weight_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_weight_generator_stack_bias_flag: bool | None = (
        config.ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG
    )
    router_bias_option_flag: bool | None = None
    router_bias_option: type[DynamicBiasConfig] | None = config.ROUTER_BIAS_OPTION
    router_bias_decay_schedule: WeightDecayScheduleOptions = (
        config.ROUTER_BIAS_DECAY_SCHEDULE
    )
    router_bias_decay_rate: float = config.ROUTER_BIAS_DECAY_RATE
    router_bias_decay_warmup_batches: int = config.ROUTER_BIAS_DECAY_WARMUP_BATCHES
    router_bias_bank_expansion_factor: BankExpansionFactorOptions = (
        config.ROUTER_BIAS_BANK_EXPANSION_FACTOR
    )
    router_bias_generator_stack_independent_flag: bool = (
        config.ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_bias_generator_stack_hidden_dim: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM
    )
    router_bias_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    router_bias_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_BIAS_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    router_bias_generator_stack_num_layers: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS
    )
    router_bias_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_ACTIVATION
    )
    router_bias_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_bias_generator_stack_residual_model_flag: bool = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_bias_generator_stack_residual_block_size: int | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_bias_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_bias_generator_stack_dropout_probability: float | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_bias_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_bias_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_bias_generator_stack_bias_flag: bool | None = (
        config.ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG
    )
    router_diagonal_option_flag: bool | None = None
    router_diagonal_option: type[DynamicDiagonalConfig] | None = (
        config.ROUTER_DIAGONAL_OPTION
    )
    router_diagonal_generator_stack_independent_flag: bool = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_diagonal_generator_stack_hidden_dim: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
    )
    router_diagonal_generator_stack_layer_norm_position: (
        LayerNormPositionOptions | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
    router_diagonal_generator_stack_normalization: NormalizationOptions | None = field(
        default=config.ROUTER_DIAGONAL_GENERATOR_STACK_NORMALIZATION, kw_only=True
    )
    router_diagonal_generator_stack_num_layers: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS
    )
    router_diagonal_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION
    )
    router_diagonal_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_diagonal_generator_stack_residual_model_flag: bool = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_diagonal_generator_stack_residual_block_size: int | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_diagonal_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_diagonal_generator_stack_dropout_probability: float | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_diagonal_generator_stack_last_layer_bias_option: (
        LastLayerBiasOptions | None
    ) = config.ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    router_diagonal_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_diagonal_generator_stack_bias_flag: bool | None = (
        config.ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG
    )
    router_mask_option_flag: bool | None = None
    router_row_mask_option: type[AxisMaskConfig] | None = config.ROUTER_ROW_MASK_OPTION
    router_mask_dimension_option: MaskDimensionOptions = (
        config.ROUTER_MASK_DIMENSION_OPTION
    )
    router_mask_threshold: float = config.ROUTER_MASK_THRESHOLD
    router_mask_surrogate_scale: float = config.ROUTER_MASK_SURROGATE_SCALE
    router_mask_floor: float = config.ROUTER_MASK_FLOOR
    router_mask_transition_width: float = config.ROUTER_MASK_TRANSITION_WIDTH
    router_mask_generator_stack_independent_flag: bool = (
        config.ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG
    )
    router_mask_generator_stack_hidden_dim: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM
    )
    router_mask_generator_stack_layer_norm_position: LayerNormPositionOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION
    )
    router_mask_generator_stack_normalization: NormalizationOptions | None = field(
        default=(config.ROUTER_MASK_GENERATOR_STACK_NORMALIZATION), kw_only=True
    )
    router_mask_generator_stack_num_layers: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS
    )
    router_mask_generator_stack_activation: ActivationOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_ACTIVATION
    )
    router_mask_generator_stack_residual_connection_option: (
        type[ResidualConfig] | None
    ) = config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
    router_mask_generator_stack_residual_model_flag: bool = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG
    )
    router_mask_generator_stack_residual_block_size: int | None = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE
    )
    router_mask_generator_stack_residual_rms_norm_epsilon: float | None = (
        config.ROUTER_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON
    )
    router_mask_generator_stack_dropout_probability: float | None = (
        config.ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
    )
    router_mask_generator_stack_last_layer_bias_option: LastLayerBiasOptions | None = (
        config.ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
    )
    router_mask_generator_stack_apply_output_postprocessing_flag: bool | None = (
        config.ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
    )
    router_mask_generator_stack_bias_flag: bool | None = (
        config.ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG
    )
    adaptive_generator_stack_options: (
        AdaptiveGeneratorStackOptions | ExpertsAdaptiveGeneratorStackOptions | None
    ) = None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None
    input_boundary_options: AdaptiveBoundaryModelOptions | None = None
    output_boundary_options: AdaptiveBoundaryModelOptions | None = None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None = None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None = None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None = None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None = None


@dataclass(frozen=True, slots=True)
class _HiddenAdaptiveDefaults:
    weight: HiddenAdaptiveWeightOptions
    bias: HiddenAdaptiveBiasOptions
    diagonal: HiddenAdaptiveDiagonalOptions
    mask: HiddenAdaptiveMaskOptions


@dataclass(frozen=True, slots=True)
class _BoundaryDefaults:
    input: AdaptiveBoundaryModelOptions
    output: AdaptiveBoundaryModelOptions


@dataclass(frozen=True, slots=True)
class _RouterAdaptiveDefaults:
    weight: HiddenAdaptiveWeightOptions
    bias: HiddenAdaptiveBiasOptions
    diagonal: HiddenAdaptiveDiagonalOptions
    mask: HiddenAdaptiveMaskOptions


@dataclass(frozen=True, slots=True)
class AdaptiveDefaults:
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    input_boundary_options: AdaptiveBoundaryModelOptions
    output_boundary_options: AdaptiveBoundaryModelOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions


@dataclass(frozen=True, slots=True)
class _GeneratorStackSourceValues:
    independent_flag: bool
    hidden_dim: int | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    num_layers: int | None
    activation: ActivationOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    residual_block_size: int | None = field(default=None, kw_only=True)
    residual_rms_norm_epsilon: float | None = field(default=None, kw_only=True)
    dropout_probability: float | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None


@dataclass(frozen=True, slots=True)
class _WeightValues:
    generator_depth: DynamicDepthOptions
    option_flag: bool
    option: type[DynamicWeightConfig] | None
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class _BiasValues:
    option_flag: bool
    option: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class _DiagonalValues:
    option_flag: bool
    option: type[DynamicDiagonalConfig] | None
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class _MaskValues:
    option_flag: bool
    row_mask_option: type[AxisMaskConfig] | None
    mask_dimension_option: MaskDimensionOptions
    mask_threshold: float
    mask_surrogate_scale: float
    mask_floor: float
    mask_transition_width: float
    generator_stack_source: AdaptiveGeneratorStackSource


@dataclass(frozen=True, slots=True)
class _BoundaryValues:
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


def _generator_stack_source(
    values: _GeneratorStackSourceValues,
) -> AdaptiveGeneratorStackSource:
    return AdaptiveGeneratorStackSource(
        independent_flag=values.independent_flag,
        hidden_dim=values.hidden_dim,
        layer_norm_position=values.layer_norm_position,
        normalization=values.normalization,
        num_layers=values.num_layers,
        activation=values.activation,
        residual_connection_option=values.residual_connection_option,
        residual_block_size=values.residual_block_size,
        residual_rms_norm_epsilon=values.residual_rms_norm_epsilon,
        residual_model_flag=values.residual_model_flag,
        dropout_probability=values.dropout_probability,
        last_layer_bias_option=values.last_layer_bias_option,
        apply_output_postprocessing_flag=values.apply_output_postprocessing_flag,
        bias_flag=values.bias_flag,
    )


def _weight_options(values: _WeightValues) -> HiddenAdaptiveWeightOptions:
    return HiddenAdaptiveWeightOptions(
        generator_depth=values.generator_depth,
        option_flag=values.option_flag,
        option=values.option,
        normalization_option=values.normalization_option,
        normalization_position_option=values.normalization_position_option,
        decay_schedule=values.decay_schedule,
        decay_rate=values.decay_rate,
        decay_warmup_batches=values.decay_warmup_batches,
        bank_expansion_factor=values.bank_expansion_factor,
        generator_stack_source=values.generator_stack_source,
    )


def _bias_options(values: _BiasValues) -> HiddenAdaptiveBiasOptions:
    return HiddenAdaptiveBiasOptions(
        option_flag=values.option_flag,
        option=values.option,
        decay_schedule=values.decay_schedule,
        decay_rate=values.decay_rate,
        decay_warmup_batches=values.decay_warmup_batches,
        bank_expansion_factor=values.bank_expansion_factor,
        generator_stack_source=values.generator_stack_source,
    )


def _diagonal_options(values: _DiagonalValues) -> HiddenAdaptiveDiagonalOptions:
    return HiddenAdaptiveDiagonalOptions(
        option_flag=values.option_flag,
        option=values.option,
        generator_stack_source=values.generator_stack_source,
    )


def _mask_options(values: _MaskValues) -> HiddenAdaptiveMaskOptions:
    return HiddenAdaptiveMaskOptions(
        option_flag=values.option_flag,
        row_mask_option=values.row_mask_option,
        mask_dimension_option=values.mask_dimension_option,
        mask_threshold=values.mask_threshold,
        mask_surrogate_scale=values.mask_surrogate_scale,
        mask_floor=values.mask_floor,
        mask_transition_width=values.mask_transition_width,
        generator_stack_source=values.generator_stack_source,
    )


def _boundary_options(values: _BoundaryValues) -> AdaptiveBoundaryModelOptions:
    return AdaptiveBoundaryModelOptions(
        weight_option=values.weight_option,
        generator_depth=values.generator_depth,
        weight_decay_schedule=values.weight_decay_schedule,
        weight_decay_rate=values.weight_decay_rate,
        weight_decay_warmup_batches=values.weight_decay_warmup_batches,
        weight_normalization_option=values.weight_normalization_option,
        weight_normalization_position_option=(
            values.weight_normalization_position_option
        ),
        weight_bank_expansion_factor=values.weight_bank_expansion_factor,
        bias_option=values.bias_option,
        bias_decay_schedule=values.bias_decay_schedule,
        bias_decay_rate=values.bias_decay_rate,
        bias_decay_warmup_batches=values.bias_decay_warmup_batches,
        bias_bank_expansion_factor=values.bias_bank_expansion_factor,
        diagonal_option=values.diagonal_option,
        row_mask_option=values.row_mask_option,
        mask_dimension_option=values.mask_dimension_option,
        mask_threshold=values.mask_threshold,
        mask_surrogate_scale=values.mask_surrogate_scale,
        mask_floor=values.mask_floor,
        mask_transition_width=values.mask_transition_width,
    )


def _adaptive_option_flag(
    explicit_flag: bool | None,
    option: type | None,
    default_flag: bool,
) -> bool:
    if explicit_flag is not None:
        return explicit_flag
    if option is not None:
        return True
    return default_flag


def _adaptive_generator_defaults(
    values: AdaptiveDefaultValues,
) -> AdaptiveGeneratorStackOptions:
    provided = values.adaptive_generator_stack_options
    if provided is None:
        return AdaptiveGeneratorStackOptions(
            hidden_dim=values.adaptive_generator_stack_hidden_dim,
            layer_norm_position=values.adaptive_generator_stack_layer_norm_position,
            normalization=values.adaptive_generator_stack_normalization,
            num_layers=values.adaptive_generator_stack_num_layers,
            activation=values.adaptive_generator_stack_activation,
            residual_connection_option=(
                values.adaptive_generator_stack_residual_connection_option
            ),
            residual_block_size=values.adaptive_generator_stack_residual_block_size,
            residual_rms_norm_epsilon=values.adaptive_generator_stack_residual_rms_norm_epsilon,
            residual_model_flag=values.adaptive_generator_stack_residual_model_flag,
            dropout_probability=values.adaptive_generator_stack_dropout_probability,
            last_layer_bias_option=(
                values.adaptive_generator_stack_last_layer_bias_option
            ),
            apply_output_postprocessing_flag=(
                values.adaptive_generator_stack_apply_output_postprocessing_flag
            ),
            bias_flag=values.adaptive_generator_stack_bias_flag,
        )
    if isinstance(provided, AdaptiveGeneratorStackOptions):
        return provided
    return AdaptiveGeneratorStackOptions(
        hidden_dim=provided.hidden_dim,
        layer_norm_position=provided.layer_norm_position,
        normalization=provided.normalization,
        num_layers=provided.num_layers,
        activation=provided.activation,
        residual_connection_option=provided.residual_connection_option,
        residual_block_size=provided.residual_block_size,
        residual_rms_norm_epsilon=provided.residual_rms_norm_epsilon,
        residual_model_flag=provided.residual_model_flag,
        dropout_probability=provided.dropout_probability,
        last_layer_bias_option=provided.last_layer_bias_option,
        apply_output_postprocessing_flag=provided.apply_output_postprocessing_flag,
        bias_flag=values.adaptive_generator_stack_bias_flag,
    )


def _hidden_adaptive_defaults(
    values: AdaptiveDefaultValues,
) -> _HiddenAdaptiveDefaults:
    hidden_adaptive_weight_options = cast(
        HiddenAdaptiveWeightOptions | None, values.hidden_adaptive_weight_options
    )
    hidden_adaptive_bias_options = cast(
        HiddenAdaptiveBiasOptions | None, values.hidden_adaptive_bias_options
    )
    hidden_adaptive_diagonal_options = cast(
        HiddenAdaptiveDiagonalOptions | None,
        values.hidden_adaptive_diagonal_options,
    )
    hidden_adaptive_mask_options = cast(
        HiddenAdaptiveMaskOptions | None, values.hidden_adaptive_mask_options
    )
    hidden_adaptive_weight_options = hidden_adaptive_weight_options or _weight_options(
        _WeightValues(
            generator_depth=values.generator_depth,
            option_flag=_adaptive_option_flag(
                values.weight_option_flag,
                values.weight_option,
                config.WEIGHT_OPTION_FLAG,
            ),
            option=values.weight_option,
            normalization_option=values.weight_normalization_option,
            normalization_position_option=(values.weight_normalization_position_option),
            decay_schedule=values.weight_decay_schedule,
            decay_rate=values.weight_decay_rate,
            decay_warmup_batches=values.weight_decay_warmup_batches,
            bank_expansion_factor=values.weight_bank_expansion_factor,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=values.weight_generator_stack_independent_flag,
                    hidden_dim=values.weight_generator_stack_hidden_dim,
                    layer_norm_position=(
                        values.weight_generator_stack_layer_norm_position
                    ),
                    normalization=(values.weight_generator_stack_normalization),
                    num_layers=values.weight_generator_stack_num_layers,
                    activation=values.weight_generator_stack_activation,
                    residual_connection_option=(
                        values.weight_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.weight_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.weight_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.weight_generator_stack_residual_model_flag,
                    dropout_probability=(
                        values.weight_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        values.weight_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.weight_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.weight_generator_stack_bias_flag,
                )
            ),
        )
    )
    hidden_adaptive_bias_options = hidden_adaptive_bias_options or _bias_options(
        _BiasValues(
            option_flag=_adaptive_option_flag(
                values.bias_option_flag,
                values.bias_option,
                config.BIAS_OPTION_FLAG,
            ),
            option=values.bias_option,
            decay_schedule=values.bias_decay_schedule,
            decay_rate=values.bias_decay_rate,
            decay_warmup_batches=values.bias_decay_warmup_batches,
            bank_expansion_factor=values.bias_bank_expansion_factor,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=values.bias_generator_stack_independent_flag,
                    hidden_dim=values.bias_generator_stack_hidden_dim,
                    layer_norm_position=values.bias_generator_stack_layer_norm_position,
                    normalization=values.bias_generator_stack_normalization,
                    num_layers=values.bias_generator_stack_num_layers,
                    activation=values.bias_generator_stack_activation,
                    residual_connection_option=(
                        values.bias_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.bias_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.bias_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.bias_generator_stack_residual_model_flag,
                    dropout_probability=values.bias_generator_stack_dropout_probability,
                    last_layer_bias_option=(
                        values.bias_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.bias_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.bias_generator_stack_bias_flag,
                )
            ),
        )
    )
    hidden_adaptive_diagonal_options = (
        hidden_adaptive_diagonal_options
        or _diagonal_options(
            _DiagonalValues(
                option_flag=_adaptive_option_flag(
                    values.diagonal_option_flag,
                    values.diagonal_option,
                    config.DIAGONAL_OPTION_FLAG,
                ),
                option=values.diagonal_option,
                generator_stack_source=_generator_stack_source(
                    _GeneratorStackSourceValues(
                        independent_flag=values.diagonal_generator_stack_independent_flag,
                        hidden_dim=values.diagonal_generator_stack_hidden_dim,
                        layer_norm_position=(
                            values.diagonal_generator_stack_layer_norm_position
                        ),
                        normalization=(values.diagonal_generator_stack_normalization),
                        num_layers=values.diagonal_generator_stack_num_layers,
                        activation=values.diagonal_generator_stack_activation,
                        residual_connection_option=(
                            values.diagonal_generator_stack_residual_connection_option
                        ),
                        residual_block_size=values.diagonal_generator_stack_residual_block_size,
                        residual_rms_norm_epsilon=values.diagonal_generator_stack_residual_rms_norm_epsilon,
                        residual_model_flag=values.diagonal_generator_stack_residual_model_flag,
                        dropout_probability=(
                            values.diagonal_generator_stack_dropout_probability
                        ),
                        last_layer_bias_option=(
                            values.diagonal_generator_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.diagonal_generator_stack_apply_output_postprocessing_flag
                        ),
                        bias_flag=values.diagonal_generator_stack_bias_flag,
                    )
                ),
            )
        )
    )
    hidden_adaptive_mask_options = hidden_adaptive_mask_options or _mask_options(
        _MaskValues(
            option_flag=_adaptive_option_flag(
                values.mask_option_flag,
                values.row_mask_option,
                config.MASK_OPTION_FLAG,
            ),
            row_mask_option=values.row_mask_option,
            mask_dimension_option=values.mask_dimension_option,
            mask_threshold=values.mask_threshold,
            mask_surrogate_scale=values.mask_surrogate_scale,
            mask_floor=values.mask_floor,
            mask_transition_width=values.mask_transition_width,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=values.mask_generator_stack_independent_flag,
                    hidden_dim=values.mask_generator_stack_hidden_dim,
                    layer_norm_position=values.mask_generator_stack_layer_norm_position,
                    normalization=values.mask_generator_stack_normalization,
                    num_layers=values.mask_generator_stack_num_layers,
                    activation=values.mask_generator_stack_activation,
                    residual_connection_option=(
                        values.mask_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.mask_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.mask_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.mask_generator_stack_residual_model_flag,
                    dropout_probability=values.mask_generator_stack_dropout_probability,
                    last_layer_bias_option=(
                        values.mask_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.mask_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.mask_generator_stack_bias_flag,
                )
            ),
        )
    )
    return _HiddenAdaptiveDefaults(
        weight=hidden_adaptive_weight_options,
        bias=hidden_adaptive_bias_options,
        diagonal=hidden_adaptive_diagonal_options,
        mask=hidden_adaptive_mask_options,
    )


def _boundary_defaults(values: AdaptiveDefaultValues) -> _BoundaryDefaults:
    input_boundary_options = cast(
        AdaptiveBoundaryModelOptions | None, values.input_boundary_options
    )
    output_boundary_options = cast(
        AdaptiveBoundaryModelOptions | None, values.output_boundary_options
    )
    input_boundary_options = input_boundary_options or _boundary_options(
        _BoundaryValues(
            weight_option=values.input_layer_weight_option,
            generator_depth=values.input_layer_generator_depth,
            weight_decay_schedule=values.input_layer_weight_decay_schedule,
            weight_decay_rate=values.input_layer_weight_decay_rate,
            weight_decay_warmup_batches=values.input_layer_weight_decay_warmup_batches,
            weight_normalization_option=values.input_layer_weight_normalization_option,
            weight_normalization_position_option=(
                values.input_layer_weight_normalization_position_option
            ),
            weight_bank_expansion_factor=values.input_layer_weight_bank_expansion_factor,
            bias_option=values.input_layer_bias_option,
            bias_decay_schedule=values.input_layer_bias_decay_schedule,
            bias_decay_rate=values.input_layer_bias_decay_rate,
            bias_decay_warmup_batches=values.input_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=values.input_layer_bias_bank_expansion_factor,
            diagonal_option=values.input_layer_diagonal_option,
            row_mask_option=values.input_layer_row_mask_option,
            mask_dimension_option=values.input_layer_mask_dimension_option,
            mask_threshold=values.input_layer_mask_threshold,
            mask_surrogate_scale=values.input_layer_mask_surrogate_scale,
            mask_floor=values.input_layer_mask_floor,
            mask_transition_width=values.input_layer_mask_transition_width,
        )
    )
    output_boundary_options = output_boundary_options or _boundary_options(
        _BoundaryValues(
            weight_option=values.output_layer_weight_option,
            generator_depth=values.output_layer_generator_depth,
            weight_decay_schedule=values.output_layer_weight_decay_schedule,
            weight_decay_rate=values.output_layer_weight_decay_rate,
            weight_decay_warmup_batches=values.output_layer_weight_decay_warmup_batches,
            weight_normalization_option=values.output_layer_weight_normalization_option,
            weight_normalization_position_option=(
                values.output_layer_weight_normalization_position_option
            ),
            weight_bank_expansion_factor=values.output_layer_weight_bank_expansion_factor,
            bias_option=values.output_layer_bias_option,
            bias_decay_schedule=values.output_layer_bias_decay_schedule,
            bias_decay_rate=values.output_layer_bias_decay_rate,
            bias_decay_warmup_batches=values.output_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=values.output_layer_bias_bank_expansion_factor,
            diagonal_option=values.output_layer_diagonal_option,
            row_mask_option=values.output_layer_row_mask_option,
            mask_dimension_option=values.output_layer_mask_dimension_option,
            mask_threshold=values.output_layer_mask_threshold,
            mask_surrogate_scale=values.output_layer_mask_surrogate_scale,
            mask_floor=values.output_layer_mask_floor,
            mask_transition_width=values.output_layer_mask_transition_width,
        )
    )
    return _BoundaryDefaults(
        input=input_boundary_options,
        output=output_boundary_options,
    )


def _router_adaptive_defaults(
    values: AdaptiveDefaultValues,
) -> _RouterAdaptiveDefaults:
    router_adaptive_weight_options = cast(
        HiddenAdaptiveWeightOptions | None, values.router_adaptive_weight_options
    )
    router_adaptive_bias_options = cast(
        HiddenAdaptiveBiasOptions | None, values.router_adaptive_bias_options
    )
    router_adaptive_diagonal_options = cast(
        HiddenAdaptiveDiagonalOptions | None,
        values.router_adaptive_diagonal_options,
    )
    router_adaptive_mask_options = cast(
        HiddenAdaptiveMaskOptions | None, values.router_adaptive_mask_options
    )
    router_adaptive_weight_options = router_adaptive_weight_options or _weight_options(
        _WeightValues(
            generator_depth=values.router_generator_depth,
            option_flag=_adaptive_option_flag(
                values.router_weight_option_flag,
                values.router_weight_option,
                config.ROUTER_WEIGHT_OPTION_FLAG,
            ),
            option=values.router_weight_option,
            normalization_option=values.router_weight_normalization_option,
            normalization_position_option=(
                values.router_weight_normalization_position_option
            ),
            decay_schedule=values.router_weight_decay_schedule,
            decay_rate=values.router_weight_decay_rate,
            decay_warmup_batches=values.router_weight_decay_warmup_batches,
            bank_expansion_factor=values.router_weight_bank_expansion_factor,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=(
                        values.router_weight_generator_stack_independent_flag
                    ),
                    hidden_dim=values.router_weight_generator_stack_hidden_dim,
                    layer_norm_position=(
                        values.router_weight_generator_stack_layer_norm_position
                    ),
                    normalization=(values.router_weight_generator_stack_normalization),
                    num_layers=values.router_weight_generator_stack_num_layers,
                    activation=values.router_weight_generator_stack_activation,
                    residual_connection_option=(
                        values.router_weight_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.router_weight_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.router_weight_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.router_weight_generator_stack_residual_model_flag,
                    dropout_probability=(
                        values.router_weight_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        values.router_weight_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_weight_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.router_weight_generator_stack_bias_flag,
                )
            ),
        )
    )
    router_adaptive_bias_options = router_adaptive_bias_options or _bias_options(
        _BiasValues(
            option_flag=_adaptive_option_flag(
                values.router_bias_option_flag,
                values.router_bias_option,
                config.ROUTER_BIAS_OPTION_FLAG,
            ),
            option=values.router_bias_option,
            decay_schedule=values.router_bias_decay_schedule,
            decay_rate=values.router_bias_decay_rate,
            decay_warmup_batches=values.router_bias_decay_warmup_batches,
            bank_expansion_factor=values.router_bias_bank_expansion_factor,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=(
                        values.router_bias_generator_stack_independent_flag
                    ),
                    hidden_dim=values.router_bias_generator_stack_hidden_dim,
                    layer_norm_position=(
                        values.router_bias_generator_stack_layer_norm_position
                    ),
                    normalization=(values.router_bias_generator_stack_normalization),
                    num_layers=values.router_bias_generator_stack_num_layers,
                    activation=values.router_bias_generator_stack_activation,
                    residual_connection_option=(
                        values.router_bias_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.router_bias_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.router_bias_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.router_bias_generator_stack_residual_model_flag,
                    dropout_probability=(
                        values.router_bias_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        values.router_bias_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_bias_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.router_bias_generator_stack_bias_flag,
                )
            ),
        )
    )
    router_adaptive_diagonal_options = (
        router_adaptive_diagonal_options
        or _diagonal_options(
            _DiagonalValues(
                option_flag=_adaptive_option_flag(
                    values.router_diagonal_option_flag,
                    values.router_diagonal_option,
                    config.ROUTER_DIAGONAL_OPTION_FLAG,
                ),
                option=values.router_diagonal_option,
                generator_stack_source=_generator_stack_source(
                    _GeneratorStackSourceValues(
                        independent_flag=(
                            values.router_diagonal_generator_stack_independent_flag
                        ),
                        hidden_dim=values.router_diagonal_generator_stack_hidden_dim,
                        layer_norm_position=(
                            values.router_diagonal_generator_stack_layer_norm_position
                        ),
                        normalization=(
                            values.router_diagonal_generator_stack_normalization
                        ),
                        num_layers=values.router_diagonal_generator_stack_num_layers,
                        activation=values.router_diagonal_generator_stack_activation,
                        residual_connection_option=(
                            values.router_diagonal_generator_stack_residual_connection_option
                        ),
                        residual_block_size=values.router_diagonal_generator_stack_residual_block_size,
                        residual_rms_norm_epsilon=values.router_diagonal_generator_stack_residual_rms_norm_epsilon,
                        residual_model_flag=values.router_diagonal_generator_stack_residual_model_flag,
                        dropout_probability=(
                            values.router_diagonal_generator_stack_dropout_probability
                        ),
                        last_layer_bias_option=(
                            values.router_diagonal_generator_stack_last_layer_bias_option
                        ),
                        apply_output_postprocessing_flag=(
                            values.router_diagonal_generator_stack_apply_output_postprocessing_flag
                        ),
                        bias_flag=values.router_diagonal_generator_stack_bias_flag,
                    )
                ),
            )
        )
    )
    router_adaptive_mask_options = router_adaptive_mask_options or _mask_options(
        _MaskValues(
            option_flag=_adaptive_option_flag(
                values.router_mask_option_flag,
                values.router_row_mask_option,
                config.ROUTER_MASK_OPTION_FLAG,
            ),
            row_mask_option=values.router_row_mask_option,
            mask_dimension_option=values.router_mask_dimension_option,
            mask_threshold=values.router_mask_threshold,
            mask_surrogate_scale=values.router_mask_surrogate_scale,
            mask_floor=values.router_mask_floor,
            mask_transition_width=values.router_mask_transition_width,
            generator_stack_source=_generator_stack_source(
                _GeneratorStackSourceValues(
                    independent_flag=values.router_mask_generator_stack_independent_flag,
                    hidden_dim=values.router_mask_generator_stack_hidden_dim,
                    layer_norm_position=(
                        values.router_mask_generator_stack_layer_norm_position
                    ),
                    normalization=(values.router_mask_generator_stack_normalization),
                    num_layers=values.router_mask_generator_stack_num_layers,
                    activation=values.router_mask_generator_stack_activation,
                    residual_connection_option=(
                        values.router_mask_generator_stack_residual_connection_option
                    ),
                    residual_block_size=values.router_mask_generator_stack_residual_block_size,
                    residual_rms_norm_epsilon=values.router_mask_generator_stack_residual_rms_norm_epsilon,
                    residual_model_flag=values.router_mask_generator_stack_residual_model_flag,
                    dropout_probability=(
                        values.router_mask_generator_stack_dropout_probability
                    ),
                    last_layer_bias_option=(
                        values.router_mask_generator_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        values.router_mask_generator_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=values.router_mask_generator_stack_bias_flag,
                )
            ),
        )
    )
    return _RouterAdaptiveDefaults(
        weight=router_adaptive_weight_options,
        bias=router_adaptive_bias_options,
        diagonal=router_adaptive_diagonal_options,
        mask=router_adaptive_mask_options,
    )


def resolve_adaptive_defaults(values: AdaptiveDefaultValues) -> AdaptiveDefaults:
    generator_stack = _adaptive_generator_defaults(values)
    hidden = _hidden_adaptive_defaults(values)
    boundary = _boundary_defaults(values)
    router = _router_adaptive_defaults(values)
    return AdaptiveDefaults(
        adaptive_generator_stack_options=generator_stack,
        hidden_adaptive_weight_options=hidden.weight,
        hidden_adaptive_bias_options=hidden.bias,
        hidden_adaptive_diagonal_options=hidden.diagonal,
        hidden_adaptive_mask_options=hidden.mask,
        input_boundary_options=boundary.input,
        output_boundary_options=boundary.output,
        router_adaptive_weight_options=router.weight,
        router_adaptive_bias_options=router.bias,
        router_adaptive_diagonal_options=router.diagonal,
        router_adaptive_mask_options=router.mask,
    )


__all__ = ["AdaptiveDefaultValues", "AdaptiveDefaults", "resolve_adaptive_defaults"]
