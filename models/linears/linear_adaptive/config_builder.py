import models.linears.linear_adaptive.config as config

from dataclasses import dataclass
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.layer.config import LayerConfig, LayerStackConfig, RecurrentLayerConfig
from models.linears.linear_adaptive.experiment_config import ExperimentConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig, LinearLayerConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
    DiagonalAxisMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


@dataclass(frozen=True)
class BoundaryLayerOptions:
    model_option: type[AdaptiveLinearLayerConfig] | None
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
    adaptive_generator_stack_num_layers: int
    adaptive_generator_stack_hidden_dim: int
    adaptive_generator_stack_activation: ActivationOptions
    adaptive_generator_stack_residual_flag: bool
    adaptive_generator_stack_dropout_probability: float
    adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions
    adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions
    adaptive_generator_stack_apply_output_pipeline_flag: bool


class LinearAdaptiveConfigBuilder:
    def __init__(
        self,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        input_dim: int = config.INPUT_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.OUTPUT_DIM,
        bias_flag: bool = config.BIAS_FLAG,
        layer_norm_position: LayerNormPositionOptions = config.STACK_LAYER_NORM_POSITION,
        stack_layer_norm_position: LayerNormPositionOptions | None = None,
        generator_depth: DynamicDepthOptions = config.WEIGHT_GENERATOR_DEPTH,
        diagonal_option: type[DynamicDiagonalConfig] | None = config.DIAGONAL_OPTION,
        bias_option: type[DynamicBiasConfig] | None = config.BIAS_OPTION,
        weight_option: type[DynamicWeightConfig] | None = config.WEIGHT_OPTION,
        weight_normalization_option: WeightNormalizationOptions = config.WEIGHT_NORMALIZATION_OPTION,
        weight_normalization_position_option: WeightNormalizationPositionOptions = config.WEIGHT_NORMALIZATION_POSITION_OPTION,
        weight_decay_schedule: WeightDecayScheduleOptions = config.WEIGHT_DECAY_SCHEDULE,
        weight_decay_rate: float = config.WEIGHT_DECAY_RATE,
        weight_decay_warmup_batches: int = config.WEIGHT_DECAY_WARMUP_BATCHES,
        weight_bank_expansion_factor: BankExpansionFactorOptions = config.WEIGHT_BANK_EXPANSION_FACTOR,
        bias_decay_schedule: WeightDecayScheduleOptions = config.BIAS_DECAY_SCHEDULE,
        bias_decay_rate: float = config.BIAS_DECAY_RATE,
        bias_decay_warmup_batches: int = config.BIAS_DECAY_WARMUP_BATCHES,
        bias_bank_expansion_factor: BankExpansionFactorOptions = config.BIAS_BANK_EXPANSION_FACTOR,
        row_mask_option: type[AxisMaskConfig] | None = config.ROW_MASK_OPTION,
        mask_dimension_option: MaskDimensionOptions = config.MASK_DIMENSION_OPTION,
        mask_threshold: float = config.MASK_THRESHOLD,
        mask_surrogate_scale: float = config.MASK_SURROGATE_SCALE,
        mask_floor: float = config.MASK_FLOOR,
        mask_transition_width: float = config.MASK_TRANSITION_WIDTH,
        stack_num_layers: int = config.STACK_NUM_LAYERS,
        stack_activation: ActivationOptions = config.STACK_ACTIVATION,
        stack_residual_flag: bool = config.STACK_RESIDUAL_FLAG,
        stack_dropout_probability: float = config.STACK_DROPOUT_PROBABILITY,
        stack_last_layer_bias_option: LastLayerBiasOptions = config.STACK_LAST_LAYER_BIAS_OPTION,
        stack_apply_output_pipeline_flag: bool = config.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        stack_gate_flag: bool = config.GATE_FLAG,
        gate_hidden_dim: int = config.GATE_HIDDEN_DIM,
        gate_layer_norm_position: LayerNormPositionOptions = config.GATE_LAYER_NORM_POSITION,
        gate_stack_num_layers: int = config.GATE_STACK_NUM_LAYERS,
        gate_stack_activation: ActivationOptions = config.GATE_STACK_ACTIVATION,
        gate_stack_residual_flag: bool = config.GATE_STACK_RESIDUAL_FLAG,
        gate_stack_dropout_probability: float = config.GATE_STACK_DROPOUT_PROBABILITY,
        gate_stack_last_layer_bias_option: LastLayerBiasOptions = config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
        gate_stack_apply_output_pipeline_flag: bool = config.GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        gate_bias_flag: bool = config.GATE_BIAS_FLAG,
        stack_halting_flag: bool = config.HALTING_FLAG,
        halting_threshold: float = config.HALTING_THRESHOLD,
        halting_dropout: float = config.HALTING_DROPOUT,
        halting_hidden_state_mode: HaltingHiddenStateModeOptions = (
            config.HALTING_HIDDEN_STATE_MODE
        ),
        halting_hidden_dim: int = config.HALTING_HIDDEN_DIM,
        halting_output_dim: int = config.HALTING_OUTPUT_DIM,
        halting_layer_norm_position: LayerNormPositionOptions = config.HALTING_LAYER_NORM_POSITION,
        halting_stack_num_layers: int = config.HALTING_STACK_NUM_LAYERS,
        halting_stack_activation: ActivationOptions = config.HALTING_STACK_ACTIVATION,
        halting_stack_residual_flag: bool = config.HALTING_STACK_RESIDUAL_FLAG,
        halting_stack_dropout_probability: float = config.HALTING_STACK_DROPOUT_PROBABILITY,
        halting_stack_last_layer_bias_option: LastLayerBiasOptions = config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        halting_stack_apply_output_pipeline_flag: bool = config.HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        halting_bias_flag: bool = config.HALTING_BIAS_FLAG,
        adaptive_generator_stack_num_layers: int = config.ADAPTIVE_STACK_NUM_LAYERS,
        adaptive_generator_stack_hidden_dim: int = config.ADAPTIVE_STACK_HIDDEN_DIM,
        adaptive_generator_stack_activation: ActivationOptions = config.ADAPTIVE_STACK_ACTIVATION,
        adaptive_generator_stack_residual_flag: bool = config.ADAPTIVE_STACK_RESIDUAL_FLAG,
        adaptive_generator_stack_dropout_probability: float = config.ADAPTIVE_STACK_DROPOUT_PROBABILITY,
        adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = config.ADAPTIVE_STACK_LAYER_NORM_POSITION,
        adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = config.ADAPTIVE_STACK_LAST_LAYER_BIAS_OPTION,
        adaptive_generator_stack_apply_output_pipeline_flag: bool = config.ADAPTIVE_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        recurrent_flag: bool = config.RECURRENT_FLAG,
        recurrent_max_steps: int = config.RECURRENT_MAX_STEPS,
        recurrent_gate_flag: bool = config.RECURRENT_GATE_FLAG,
        recurrent_halting_flag: bool = config.RECURRENT_HALTING_FLAG,
        input_layer_model_option: type[AdaptiveLinearLayerConfig] | None = (
            config.INPUT_LAYER_MODEL_OPTION
        ),
        input_layer_weight_option: type[DynamicWeightConfig] | None = (
            config.INPUT_LAYER_WEIGHT_OPTION
        ),
        input_layer_weight_generator_depth: DynamicDepthOptions = (
            config.INPUT_LAYER_WEIGHT_GENERATOR_DEPTH
        ),
        input_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
            config.INPUT_LAYER_WEIGHT_DECAY_SCHEDULE
        ),
        input_layer_weight_decay_rate: float = config.INPUT_LAYER_WEIGHT_DECAY_RATE,
        input_layer_weight_decay_warmup_batches: int = (
            config.INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
        ),
        input_layer_weight_normalization_option: WeightNormalizationOptions = (
            config.INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
        ),
        input_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = (
            config.INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
        ),
        input_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
            config.INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
        ),
        input_layer_bias_option: type[DynamicBiasConfig] | None = (
            config.INPUT_LAYER_BIAS_OPTION
        ),
        input_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
            config.INPUT_LAYER_BIAS_DECAY_SCHEDULE
        ),
        input_layer_bias_decay_rate: float = config.INPUT_LAYER_BIAS_DECAY_RATE,
        input_layer_bias_decay_warmup_batches: int = (
            config.INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
        ),
        input_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
            config.INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
        ),
        input_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
            config.INPUT_LAYER_DIAGONAL_OPTION
        ),
        input_layer_row_mask_option: type[AxisMaskConfig] | None = (
            config.INPUT_LAYER_ROW_MASK_OPTION
        ),
        input_layer_mask_dimension_option: MaskDimensionOptions = (
            config.INPUT_LAYER_MASK_DIMENSION_OPTION
        ),
        input_layer_mask_threshold: float = config.INPUT_LAYER_MASK_THRESHOLD,
        input_layer_mask_surrogate_scale: float = (
            config.INPUT_LAYER_MASK_SURROGATE_SCALE
        ),
        input_layer_mask_floor: float = config.INPUT_LAYER_MASK_FLOOR,
        input_layer_mask_transition_width: float = (
            config.INPUT_LAYER_MASK_TRANSITION_WIDTH
        ),
        input_layer_adaptive_generator_stack_num_layers: int = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS
        ),
        input_layer_adaptive_generator_stack_hidden_dim: int = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM
        ),
        input_layer_adaptive_generator_stack_activation: ActivationOptions = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_ACTIVATION
        ),
        input_layer_adaptive_generator_stack_residual_flag: bool = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG
        ),
        input_layer_adaptive_generator_stack_dropout_probability: float = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY
        ),
        input_layer_adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION
        ),
        input_layer_adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
        ),
        input_layer_adaptive_generator_stack_apply_output_pipeline_flag: bool = (
            config.INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        output_layer_model_option: type[AdaptiveLinearLayerConfig] | None = (
            config.OUTPUT_LAYER_MODEL_OPTION
        ),
        output_layer_weight_option: type[DynamicWeightConfig] | None = (
            config.OUTPUT_LAYER_WEIGHT_OPTION
        ),
        output_layer_weight_generator_depth: DynamicDepthOptions = (
            config.OUTPUT_LAYER_WEIGHT_GENERATOR_DEPTH
        ),
        output_layer_weight_decay_schedule: WeightDecayScheduleOptions = (
            config.OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE
        ),
        output_layer_weight_decay_rate: float = config.OUTPUT_LAYER_WEIGHT_DECAY_RATE,
        output_layer_weight_decay_warmup_batches: int = (
            config.OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES
        ),
        output_layer_weight_normalization_option: WeightNormalizationOptions = (
            config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION
        ),
        output_layer_weight_normalization_position_option: WeightNormalizationPositionOptions = (
            config.OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION
        ),
        output_layer_weight_bank_expansion_factor: BankExpansionFactorOptions = (
            config.OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR
        ),
        output_layer_bias_option: type[DynamicBiasConfig] | None = (
            config.OUTPUT_LAYER_BIAS_OPTION
        ),
        output_layer_bias_decay_schedule: WeightDecayScheduleOptions = (
            config.OUTPUT_LAYER_BIAS_DECAY_SCHEDULE
        ),
        output_layer_bias_decay_rate: float = config.OUTPUT_LAYER_BIAS_DECAY_RATE,
        output_layer_bias_decay_warmup_batches: int = (
            config.OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES
        ),
        output_layer_bias_bank_expansion_factor: BankExpansionFactorOptions = (
            config.OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR
        ),
        output_layer_diagonal_option: type[DynamicDiagonalConfig] | None = (
            config.OUTPUT_LAYER_DIAGONAL_OPTION
        ),
        output_layer_row_mask_option: type[AxisMaskConfig] | None = (
            config.OUTPUT_LAYER_ROW_MASK_OPTION
        ),
        output_layer_mask_dimension_option: MaskDimensionOptions = (
            config.OUTPUT_LAYER_MASK_DIMENSION_OPTION
        ),
        output_layer_mask_threshold: float = config.OUTPUT_LAYER_MASK_THRESHOLD,
        output_layer_mask_surrogate_scale: float = (
            config.OUTPUT_LAYER_MASK_SURROGATE_SCALE
        ),
        output_layer_mask_floor: float = config.OUTPUT_LAYER_MASK_FLOOR,
        output_layer_mask_transition_width: float = (
            config.OUTPUT_LAYER_MASK_TRANSITION_WIDTH
        ),
        output_layer_adaptive_generator_stack_num_layers: int = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS
        ),
        output_layer_adaptive_generator_stack_hidden_dim: int = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM
        ),
        output_layer_adaptive_generator_stack_activation: ActivationOptions = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_ACTIVATION
        ),
        output_layer_adaptive_generator_stack_residual_flag: bool = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG
        ),
        output_layer_adaptive_generator_stack_dropout_probability: float = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY
        ),
        output_layer_adaptive_generator_stack_layer_norm_position: LayerNormPositionOptions = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION
        ),
        output_layer_adaptive_generator_stack_last_layer_bias_option: LastLayerBiasOptions = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
        ),
        output_layer_adaptive_generator_stack_apply_output_pipeline_flag: bool = (
            config.OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias_flag = bias_flag
        self.layer_norm_position = layer_norm_position
        self.stack_layer_norm_position = stack_layer_norm_position
        self.generator_depth = generator_depth
        self.diagonal_option = diagonal_option
        self.bias_option = bias_option
        self.weight_option = weight_option
        self.weight_normalization_option = weight_normalization_option
        self.weight_normalization_position_option = weight_normalization_position_option
        self.weight_decay_schedule = weight_decay_schedule
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_warmup_batches = weight_decay_warmup_batches
        self.weight_bank_expansion_factor = weight_bank_expansion_factor
        self.bias_decay_schedule = bias_decay_schedule
        self.bias_decay_rate = bias_decay_rate
        self.bias_decay_warmup_batches = bias_decay_warmup_batches
        self.bias_bank_expansion_factor = bias_bank_expansion_factor
        self.row_mask_option = row_mask_option
        self.mask_dimension_option = mask_dimension_option
        self.mask_threshold = mask_threshold
        self.mask_surrogate_scale = mask_surrogate_scale
        self.mask_floor = mask_floor
        self.mask_transition_width = mask_transition_width
        self.stack_num_layers = stack_num_layers
        self.stack_activation = stack_activation
        self.stack_residual_flag = stack_residual_flag
        self.stack_dropout_probability = stack_dropout_probability
        self.stack_last_layer_bias_option = stack_last_layer_bias_option
        self.stack_apply_output_pipeline_flag = stack_apply_output_pipeline_flag
        self.stack_gate_flag = stack_gate_flag
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_layer_norm_position = gate_layer_norm_position
        self.gate_stack_num_layers = gate_stack_num_layers
        self.gate_stack_activation = gate_stack_activation
        self.gate_stack_residual_flag = gate_stack_residual_flag
        self.gate_stack_dropout_probability = gate_stack_dropout_probability
        self.gate_stack_last_layer_bias_option = gate_stack_last_layer_bias_option
        self.gate_stack_apply_output_pipeline_flag = (
            gate_stack_apply_output_pipeline_flag
        )
        self.gate_bias_flag = gate_bias_flag
        self.stack_halting_flag = stack_halting_flag
        self.halting_threshold = halting_threshold
        self.halting_dropout = halting_dropout
        self.halting_hidden_state_mode = halting_hidden_state_mode
        self.halting_hidden_dim = halting_hidden_dim
        self.halting_output_dim = halting_output_dim
        self.halting_layer_norm_position = halting_layer_norm_position
        self.halting_stack_num_layers = halting_stack_num_layers
        self.halting_stack_activation = halting_stack_activation
        self.halting_stack_residual_flag = halting_stack_residual_flag
        self.halting_stack_dropout_probability = halting_stack_dropout_probability
        self.halting_stack_last_layer_bias_option = halting_stack_last_layer_bias_option
        self.halting_stack_apply_output_pipeline_flag = (
            halting_stack_apply_output_pipeline_flag
        )
        self.halting_bias_flag = halting_bias_flag
        self.adaptive_generator_stack_num_layers = adaptive_generator_stack_num_layers
        self.adaptive_generator_stack_hidden_dim = adaptive_generator_stack_hidden_dim
        self.adaptive_generator_stack_activation = adaptive_generator_stack_activation
        self.adaptive_generator_stack_residual_flag = (
            adaptive_generator_stack_residual_flag
        )
        self.adaptive_generator_stack_dropout_probability = (
            adaptive_generator_stack_dropout_probability
        )
        self.adaptive_generator_stack_layer_norm_position = (
            adaptive_generator_stack_layer_norm_position
        )
        self.adaptive_generator_stack_last_layer_bias_option = (
            adaptive_generator_stack_last_layer_bias_option
        )
        self.adaptive_generator_stack_apply_output_pipeline_flag = (
            adaptive_generator_stack_apply_output_pipeline_flag
        )
        self.recurrent_flag = recurrent_flag
        self.recurrent_max_steps = recurrent_max_steps
        self.recurrent_gate_flag = recurrent_gate_flag
        self.recurrent_halting_flag = recurrent_halting_flag
        self.input_boundary_options = BoundaryLayerOptions(
            model_option=input_layer_model_option,
            weight_option=input_layer_weight_option,
            weight_generator_depth=input_layer_weight_generator_depth,
            weight_decay_schedule=input_layer_weight_decay_schedule,
            weight_decay_rate=input_layer_weight_decay_rate,
            weight_decay_warmup_batches=input_layer_weight_decay_warmup_batches,
            weight_normalization_option=input_layer_weight_normalization_option,
            weight_normalization_position_option=(
                input_layer_weight_normalization_position_option
            ),
            weight_bank_expansion_factor=input_layer_weight_bank_expansion_factor,
            bias_option=input_layer_bias_option,
            bias_decay_schedule=input_layer_bias_decay_schedule,
            bias_decay_rate=input_layer_bias_decay_rate,
            bias_decay_warmup_batches=input_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=input_layer_bias_bank_expansion_factor,
            diagonal_option=input_layer_diagonal_option,
            row_mask_option=input_layer_row_mask_option,
            mask_dimension_option=input_layer_mask_dimension_option,
            mask_threshold=input_layer_mask_threshold,
            mask_surrogate_scale=input_layer_mask_surrogate_scale,
            mask_floor=input_layer_mask_floor,
            mask_transition_width=input_layer_mask_transition_width,
            adaptive_generator_stack_num_layers=(
                input_layer_adaptive_generator_stack_num_layers
            ),
            adaptive_generator_stack_hidden_dim=(
                input_layer_adaptive_generator_stack_hidden_dim
            ),
            adaptive_generator_stack_activation=(
                input_layer_adaptive_generator_stack_activation
            ),
            adaptive_generator_stack_residual_flag=(
                input_layer_adaptive_generator_stack_residual_flag
            ),
            adaptive_generator_stack_dropout_probability=(
                input_layer_adaptive_generator_stack_dropout_probability
            ),
            adaptive_generator_stack_layer_norm_position=(
                input_layer_adaptive_generator_stack_layer_norm_position
            ),
            adaptive_generator_stack_last_layer_bias_option=(
                input_layer_adaptive_generator_stack_last_layer_bias_option
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=(
                input_layer_adaptive_generator_stack_apply_output_pipeline_flag
            ),
        )
        self.output_boundary_options = BoundaryLayerOptions(
            model_option=output_layer_model_option,
            weight_option=output_layer_weight_option,
            weight_generator_depth=output_layer_weight_generator_depth,
            weight_decay_schedule=output_layer_weight_decay_schedule,
            weight_decay_rate=output_layer_weight_decay_rate,
            weight_decay_warmup_batches=output_layer_weight_decay_warmup_batches,
            weight_normalization_option=output_layer_weight_normalization_option,
            weight_normalization_position_option=(
                output_layer_weight_normalization_position_option
            ),
            weight_bank_expansion_factor=output_layer_weight_bank_expansion_factor,
            bias_option=output_layer_bias_option,
            bias_decay_schedule=output_layer_bias_decay_schedule,
            bias_decay_rate=output_layer_bias_decay_rate,
            bias_decay_warmup_batches=output_layer_bias_decay_warmup_batches,
            bias_bank_expansion_factor=output_layer_bias_bank_expansion_factor,
            diagonal_option=output_layer_diagonal_option,
            row_mask_option=output_layer_row_mask_option,
            mask_dimension_option=output_layer_mask_dimension_option,
            mask_threshold=output_layer_mask_threshold,
            mask_surrogate_scale=output_layer_mask_surrogate_scale,
            mask_floor=output_layer_mask_floor,
            mask_transition_width=output_layer_mask_transition_width,
            adaptive_generator_stack_num_layers=(
                output_layer_adaptive_generator_stack_num_layers
            ),
            adaptive_generator_stack_hidden_dim=(
                output_layer_adaptive_generator_stack_hidden_dim
            ),
            adaptive_generator_stack_activation=(
                output_layer_adaptive_generator_stack_activation
            ),
            adaptive_generator_stack_residual_flag=(
                output_layer_adaptive_generator_stack_residual_flag
            ),
            adaptive_generator_stack_dropout_probability=(
                output_layer_adaptive_generator_stack_dropout_probability
            ),
            adaptive_generator_stack_layer_norm_position=(
                output_layer_adaptive_generator_stack_layer_norm_position
            ),
            adaptive_generator_stack_last_layer_bias_option=(
                output_layer_adaptive_generator_stack_last_layer_bias_option
            ),
            adaptive_generator_stack_apply_output_pipeline_flag=(
                output_layer_adaptive_generator_stack_apply_output_pipeline_flag
            ),
        )

    def build(self) -> "ModelConfig":
        from emperor.config import ModelConfig

        input_model_config = self._build_boundary_layer_config(
            boundary_name="input",
            options=self.input_boundary_options,
            activation=self.stack_activation,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.stack_dropout_probability,
        )

        gate_config = self._build_gate_config()
        halting_config = self._build_halting_config()
        adaptive_weight_config = self._build_weight_config()
        adaptive_bias_config = self._build_bias_config()
        adaptive_diagonal_config = self._build_diagonal_config()
        adaptive_mask_config = self._build_mask_config()
        adaptive_model_config = self._build_model_config()
        model_config = LayerStackConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.stack_num_layers,
            last_layer_bias_option=self.stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.stack_activation,
                layer_norm_position=self.layer_norm_position,
                residual_flag=self.stack_residual_flag,
                dropout_probability=self.stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                shared_halting_flag=False,
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=self.bias_flag,
                    adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                        weight_config=adaptive_weight_config,
                        bias_config=adaptive_bias_config,
                        diagonal_config=adaptive_diagonal_config,
                        mask_config=adaptive_mask_config,
                        model_config=adaptive_model_config,
                    ),
                ),
            ),
        )
        model_config = self._maybe_wrap_recurrent(model_config)

        output_model_config = self._build_boundary_layer_config(
            boundary_name="output",
            options=self.output_boundary_options,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
        )

        return ModelConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            experiment_config=ExperimentConfig(
                input_model_config=input_model_config,
                model_config=model_config,
                output_model_config=output_model_config,
            ),
        )

    def _maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> "LayerStackConfig | RecurrentLayerConfig":
        if not self.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_max_steps,
            block_config=block_config,
            gate_config=self._build_gate_config(self.recurrent_gate_flag),
            halting_config=self._build_halting_config(self.recurrent_halting_flag),
        )

    def _build_boundary_layer_config(
        self,
        boundary_name: str,
        options: BoundaryLayerOptions,
        activation: ActivationOptions,
        layer_norm_position: LayerNormPositionOptions,
        dropout_probability: float,
    ) -> LayerConfig:
        return LayerConfig(
            activation=activation,
            layer_norm_position=layer_norm_position,
            residual_flag=False,
            dropout_probability=dropout_probability,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=self._build_boundary_layer_model_config(
                boundary_name,
                options,
            ),
        )

    def _build_boundary_layer_model_config(
        self,
        boundary_name: str,
        options: BoundaryLayerOptions,
    ) -> LinearLayerConfig:
        self._validate_boundary_layer_options(boundary_name, options)
        if options.model_option is None:
            return LinearLayerConfig(bias_flag=self.bias_flag)
        return AdaptiveLinearLayerConfig(
            bias_flag=self.bias_flag,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=self._build_weight_config_from_options(
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
                bias_config=self._build_bias_config_from_options(
                    bias_option=options.bias_option,
                    decay_schedule=options.bias_decay_schedule,
                    decay_rate=options.bias_decay_rate,
                    decay_warmup_batches=options.bias_decay_warmup_batches,
                    bank_expansion_factor=options.bias_bank_expansion_factor,
                ),
                diagonal_config=self._build_diagonal_config_from_option(
                    options.diagonal_option
                ),
                mask_config=self._build_mask_config_from_options(
                    row_mask_option=options.row_mask_option,
                    mask_dimension_option=options.mask_dimension_option,
                    mask_threshold=options.mask_threshold,
                    mask_surrogate_scale=options.mask_surrogate_scale,
                    mask_floor=options.mask_floor,
                    mask_transition_width=options.mask_transition_width,
                ),
                model_config=self._build_model_config_from_options(
                    hidden_dim=options.adaptive_generator_stack_hidden_dim,
                    num_layers=options.adaptive_generator_stack_num_layers,
                    activation=options.adaptive_generator_stack_activation,
                    residual_flag=options.adaptive_generator_stack_residual_flag,
                    dropout_probability=(
                        options.adaptive_generator_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        options.adaptive_generator_stack_layer_norm_position
                    ),
                    last_layer_bias_option=(
                        options.adaptive_generator_stack_last_layer_bias_option
                    ),
                    apply_output_pipeline_flag=(
                        options.adaptive_generator_stack_apply_output_pipeline_flag
                    ),
                ),
            ),
        )

    def _validate_boundary_layer_options(
        self,
        boundary_name: str,
        options: BoundaryLayerOptions,
    ) -> None:
        if options.model_option not in {None, AdaptiveLinearLayerConfig}:
            received_option = getattr(
                options.model_option, "__name__", repr(options.model_option)
            )
            raise ValueError(
                f"{boundary_name}_layer_model_option must be None or "
                f"AdaptiveLinearLayerConfig, received "
                f"{received_option}."
            )
        if options.model_option is not None:
            return

        adaptive_fields = {
            f"{boundary_name}_layer_weight_option": options.weight_option,
            f"{boundary_name}_layer_bias_option": options.bias_option,
            f"{boundary_name}_layer_diagonal_option": options.diagonal_option,
            f"{boundary_name}_layer_row_mask_option": options.row_mask_option,
        }
        enabled_fields = [
            field_name
            for field_name, field_value in adaptive_fields.items()
            if field_value is not None
        ]
        if enabled_fields:
            raise ValueError(
                f"{boundary_name} boundary projector adaptive options require "
                f"{boundary_name}_layer_model_option=AdaptiveLinearLayerConfig; "
                f"received {', '.join(enabled_fields)} with a default linear "
                "projector."
            )

    def _build_gate_config(self, enabled: bool | None = None) -> LayerStackConfig | None:
        if enabled is None:
            enabled = self.stack_gate_flag
        if not enabled:
            return None
        return LayerStackConfig(
            hidden_dim=self.gate_hidden_dim,
            num_layers=self.gate_stack_num_layers,
            last_layer_bias_option=self.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=self.gate_stack_apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=self.gate_stack_activation,
                layer_norm_position=self.gate_layer_norm_position,
                residual_flag=self.gate_stack_residual_flag,
                dropout_probability=self.gate_stack_dropout_probability,
                halting_config=None,
                shared_halting_flag=False,
                gate_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.gate_bias_flag,
                ),
            ),
        )

    def _build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        if enabled is None:
            enabled = self.stack_halting_flag
        if not enabled:
            return None
        return StickBreakingConfig(
            threshold=self.halting_threshold,
            halting_dropout=self.halting_dropout,
            hidden_state_mode=self.halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                hidden_dim=self.halting_hidden_dim or self.output_dim,
                output_dim=self.halting_output_dim,
                num_layers=self.halting_stack_num_layers,
                last_layer_bias_option=self.halting_stack_last_layer_bias_option,
                apply_output_pipeline_flag=self.halting_stack_apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=self.halting_stack_activation,
                    layer_norm_position=self.halting_layer_norm_position,
                    residual_flag=self.halting_stack_residual_flag,
                    dropout_probability=self.halting_stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.halting_bias_flag,
                    ),
                ),
            ),
        )

    def _build_weight_config(self) -> DynamicWeightConfig | None:
        return self._build_weight_config_from_options(
            weight_option=self.weight_option,
            generator_depth=self.generator_depth,
            decay_schedule=self.weight_decay_schedule,
            decay_rate=self.weight_decay_rate,
            decay_warmup_batches=self.weight_decay_warmup_batches,
            normalization_option=self.weight_normalization_option,
            normalization_position_option=self.weight_normalization_position_option,
            bank_expansion_factor=self.weight_bank_expansion_factor,
        )

    def _build_weight_config_from_options(
        self,
        weight_option: type[DynamicWeightConfig] | None,
        generator_depth: DynamicDepthOptions,
        decay_schedule: WeightDecayScheduleOptions,
        decay_rate: float,
        decay_warmup_batches: int,
        normalization_option: WeightNormalizationOptions,
        normalization_position_option: WeightNormalizationPositionOptions,
        bank_expansion_factor: BankExpansionFactorOptions,
    ) -> DynamicWeightConfig | None:
        if weight_option is None:
            return None
        kwargs: dict[str, Any] = {
            "generator_depth": generator_depth,
            "decay_schedule": decay_schedule,
            "decay_rate": decay_rate,
            "decay_warmup_batches": decay_warmup_batches,
        }
        if weight_option in {
            SingleModelDynamicWeightConfig,
            DualModelDynamicWeightConfig,
        }:
            kwargs["normalization_option"] = normalization_option
            kwargs["normalization_position_option"] = (
                normalization_position_option
            )
        elif weight_option in {
            LowRankDynamicWeightConfig,
            HypernetworkDynamicWeightConfig,
        }:
            kwargs["normalization_option"] = normalization_option
        elif weight_option in {
            LayeredWeightedBankDynamicWeightConfig,
            SoftWeightedBankDynamicWeightConfig,
        }:
            kwargs["bank_expansion_factor"] = bank_expansion_factor
        return weight_option(**kwargs)

    def _build_bias_config(self) -> DynamicBiasConfig | None:
        return self._build_bias_config_from_options(
            bias_option=self.bias_option,
            decay_schedule=self.bias_decay_schedule,
            decay_rate=self.bias_decay_rate,
            decay_warmup_batches=self.bias_decay_warmup_batches,
            bank_expansion_factor=self.bias_bank_expansion_factor,
        )

    def _build_bias_config_from_options(
        self,
        bias_option: type[DynamicBiasConfig] | None,
        decay_schedule: WeightDecayScheduleOptions,
        decay_rate: float,
        decay_warmup_batches: int,
        bank_expansion_factor: BankExpansionFactorOptions,
    ) -> DynamicBiasConfig | None:
        if bias_option is None:
            return None
        kwargs: dict[str, Any] = {
            "decay_schedule": decay_schedule,
            "decay_rate": decay_rate,
            "decay_warmup_batches": decay_warmup_batches,
        }
        if bias_option is WeightedBankDynamicBiasConfig:
            kwargs["bank_expansion_factor"] = bank_expansion_factor
        return bias_option(**kwargs)

    def _build_diagonal_config(self) -> DynamicDiagonalConfig | None:
        return self._build_diagonal_config_from_option(self.diagonal_option)

    def _build_diagonal_config_from_option(
        self,
        diagonal_option: type[DynamicDiagonalConfig] | None,
    ) -> DynamicDiagonalConfig | None:
        if diagonal_option is None:
            return None
        return diagonal_option()

    def _build_mask_config(self) -> AxisMaskConfig | None:
        return self._build_mask_config_from_options(
            row_mask_option=self.row_mask_option,
            mask_dimension_option=self.mask_dimension_option,
            mask_threshold=self.mask_threshold,
            mask_surrogate_scale=self.mask_surrogate_scale,
            mask_floor=self.mask_floor,
            mask_transition_width=self.mask_transition_width,
        )

    def _build_mask_config_from_options(
        self,
        row_mask_option: type[AxisMaskConfig] | None,
        mask_dimension_option: MaskDimensionOptions,
        mask_threshold: float,
        mask_surrogate_scale: float,
        mask_floor: float,
        mask_transition_width: float,
    ) -> AxisMaskConfig | None:
        if row_mask_option is None:
            return None
        kwargs: dict[str, Any] = {
            "mask_threshold": mask_threshold,
            "mask_surrogate_scale": mask_surrogate_scale,
            "mask_floor": mask_floor,
        }
        if row_mask_option in {
            WeightInformedScoreAxisMaskConfig,
            PerAxisScoreMaskConfig,
            TopSliceAxisMaskConfig,
        }:
            kwargs["mask_dimension_option"] = mask_dimension_option
        if row_mask_option in {TopSliceAxisMaskConfig, DiagonalAxisMaskConfig}:
            kwargs["mask_transition_width"] = mask_transition_width
        return row_mask_option(**kwargs)

    def _build_model_config(self) -> LayerStackConfig:
        return self._build_model_config_from_options(
            hidden_dim=self.adaptive_generator_stack_hidden_dim,
            num_layers=self.adaptive_generator_stack_num_layers,
            activation=self.adaptive_generator_stack_activation,
            residual_flag=self.adaptive_generator_stack_residual_flag,
            dropout_probability=self.adaptive_generator_stack_dropout_probability,
            layer_norm_position=self.adaptive_generator_stack_layer_norm_position,
            last_layer_bias_option=self.adaptive_generator_stack_last_layer_bias_option,
            apply_output_pipeline_flag=(
                self.adaptive_generator_stack_apply_output_pipeline_flag
            ),
        )

    def _build_model_config_from_options(
        self,
        hidden_dim: int,
        num_layers: int,
        activation: ActivationOptions,
        residual_flag: bool,
        dropout_probability: float,
        layer_norm_position: LayerNormPositionOptions,
        last_layer_bias_option: LastLayerBiasOptions,
        apply_output_pipeline_flag: bool,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=activation,
                layer_norm_position=layer_norm_position,
                residual_flag=residual_flag,
                dropout_probability=dropout_probability,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    bias_flag=self.bias_flag,
                ),
            ),
        )
