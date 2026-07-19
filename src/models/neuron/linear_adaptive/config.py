# ruff: noqa: E402, F401, E501

from emperor.layers import ResidualConnectionOptions

# Trainer
TRAINER_ACCELERATOR: str = "auto"
TRAINER_DEVICES: str | int = "auto"
TRAINER_GRADIENT_CLIP_VAL: float = 0.0
TRAINER_GRADIENT_CLIP_ALGORITHM: str = "norm"
TRAINER_ACCUMULATE_GRAD_BATCHES: int = 1
TRAINER_PRECISION: str = "32-true"
TRAINER_DETERMINISTIC: bool = False
TRAINER_BENCHMARK: bool = True
TRAINER_MAX_STEPS: int = -1
TRAINER_MAX_TIME: str | None = None
TRAINER_VAL_CHECK_INTERVAL: float = 1.0
TRAINER_LIMIT_TRAIN_BATCHES: float = 1.0
TRAINER_LIMIT_VAL_BATCHES: float = 1.0
TRAINER_OVERFIT_BATCHES: int | float = 0.0
TRAINER_NUM_SANITY_VAL_STEPS: int = 2
TRAINER_LOG_EVERY_N_STEPS: int = 50
TRAINER_ENABLE_PROGRESS_BAR: bool = False
TRAINER_ENABLE_CHECKPOINTING: bool = False
TRAINER_ENABLE_MODEL_SUMMARY: bool = False
TRAINER_PROFILER: str | None = None
MONITOR_LOG_EVERY_N_STEPS: int = 100

# Run
DATA_NUM_WORKERS: int = 4
RUN_TEST_AFTER_FIT: bool = True

# Callback
CALLBACK_EARLY_STOPPING_PATIENCE: int = 0
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"
CALLBACK_EARLY_STOPPING_MIN_DELTA: float = 0.0
CALLBACK_EARLY_STOPPING_STRICT: bool = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: bool = True
CALLBACK_CHECKPOINT_FLAG: bool = False
from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    BankExpansionFactorOptions,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MaskDimensionOptions,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)

# Global
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 10
INPUT_DIM: int = 28**2
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 10
LEARNING_RATE: float = 1e-3

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0


#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
STACK_DROPOUT_PROBABILITY: float = 0.0
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True

#########################################################################
# Layer Stack Submodule Options
SUBMODULE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    STACK_LAYER_NORM_POSITION
)
SUBMODULE_STACK_NUM_LAYERS: int = 2
SUBMODULE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
SUBMODULE_STACK_DROPOUT_PROBABILITY: float = 0.0
SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
SUBMODULE_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# Adaptive Generator Stack Options
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    STACK_LAYER_NORM_POSITION
)
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# Gate Options
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Gate Stack Options
# If False, gate model stack options inherit the layer stack submodule options.
GATE_STACK_INDEPENDENT_FLAG: bool = False
GATE_STACK_HIDDEN_DIM: int | None = None
GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
GATE_STACK_NUM_LAYERS: int | None = None
GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
GATE_STACK_DROPOUT_PROBABILITY: float | None = None
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
GATE_STACK_BIAS_FLAG: bool | None = True

#########################################################################
# Halting Options
# If `HALTING_FLAG` is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
HALTING_THRESHOLD: float = 0.999
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
## Halting Stack Options
# If False, halting model stack options inherit the layer stack submodule options.
HALTING_STACK_INDEPENDENT_FLAG: bool = False
HALTING_STACK_HIDDEN_DIM: int | None = None
HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
HALTING_STACK_NUM_LAYERS: int | None = None
HALTING_STACK_ACTIVATION: ActivationOptions | None = None
HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Memory Options
# If `MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
MEMORY_FLAG: bool = False
MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Memory Stack Options
# If False, memory model stack options inherit the layer stack submodule options.
MEMORY_STACK_INDEPENDENT_FLAG: bool = False
MEMORY_STACK_HIDDEN_DIM: int | None = None
MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
MEMORY_STACK_NUM_LAYERS: int | None = None
MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Recurrent Layer Options
# If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

#########################################################################
## Recurrent Gate Options
RECURRENT_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
### Recurrent Gate Stack Options
# If False, recurrent gate stack options inherit gate/submodule stack options.
RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Recurrent Halting Options
RECURRENT_HALTING_FLAG: bool = False
RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
RECURRENT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
RECURRENT_HALTING_DROPOUT: float = HALTING_DROPOUT
RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
### Recurrent Halting Stack Options
# If False, recurrent halting stack options inherit halting/submodule stack options.
RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Weight Generator Options
# If `WEIGHT_OPTION_FLAG` is False, the weight-specific parameters below are ignored.
WEIGHT_OPTION_FLAG: bool = False
WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
WEIGHT_DECAY_RATE: float = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: int = 0
WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WeightNormalizationOptions.DISABLED
)
WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
)
WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_THREE
)
## Weight Generator Stack Options
# If False, weight generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
WEIGHT_GENERATOR_STACK_HIDDEN_DIM: int | None = None
WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
WEIGHT_GENERATOR_STACK_BIAS_FLAG: bool | None = None


#########################################################################
# Bias Generator Options
# If `BIAS_OPTION_FLAG` is False, the bias-specific parameters below are ignored.
BIAS_OPTION_FLAG: bool = False
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)
## Bias Generator Stack Options
# If False, bias generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
BIAS_GENERATOR_STACK_HIDDEN_DIM: int | None = None
BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = None
BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
BIAS_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Diagonal Generator Options
# If `DIAGONAL_OPTION_FLAG` is False, the diagonal-specific parameters below are
# ignored.
DIAGONAL_OPTION_FLAG: bool = False
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
## Diagonal Generator Stack Options
# If False, diagonal generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: int | None = None
DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = None
DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
DIAGONAL_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Mask Options
# If `MASK_OPTION_FLAG` is False, the mask-specific parameters below are ignored.
MASK_OPTION_FLAG: bool = False
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1
MASK_SURROGATE_SCALE: float = 10.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN
## Mask Stack Options
# If False, mask generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
MASK_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
MASK_GENERATOR_STACK_HIDDEN_DIM: int | None = None
MASK_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
MASK_GENERATOR_STACK_NUM_LAYERS: int | None = None
MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = None


#########################################################################
# Input Boundary Model Options
# Input boundary dynamic weight options.
INPUT_LAYER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
INPUT_LAYER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
INPUT_LAYER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
INPUT_LAYER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WEIGHT_NORMALIZATION_POSITION_OPTION
)
INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
# Input boundary dynamic bias options.
INPUT_LAYER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
INPUT_LAYER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
INPUT_LAYER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
# Input boundary dynamic diagonal options.
INPUT_LAYER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# Input boundary dynamic mask options.
INPUT_LAYER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
INPUT_LAYER_MASK_THRESHOLD: float = MASK_THRESHOLD
INPUT_LAYER_MASK_FLOOR: float = MASK_FLOOR
INPUT_LAYER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
INPUT_LAYER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
INPUT_LAYER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION

#########################################################################
# Output Boundary Model Options
# Output boundary dynamic weight options.
OUTPUT_LAYER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
OUTPUT_LAYER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
OUTPUT_LAYER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = WEIGHT_NORMALIZATION_POSITION_OPTION
OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
# Output boundary dynamic bias options.
OUTPUT_LAYER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
OUTPUT_LAYER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
OUTPUT_LAYER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
# Output boundary dynamic diagonal options.
OUTPUT_LAYER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# Output boundary dynamic mask options.
OUTPUT_LAYER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
OUTPUT_LAYER_MASK_THRESHOLD: float = MASK_THRESHOLD
OUTPUT_LAYER_MASK_FLOOR: float = MASK_FLOOR
OUTPUT_LAYER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
OUTPUT_LAYER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
OUTPUT_LAYER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION

#########################################################################
# Neuron Wrapper Options
from emperor.neuron import (
    NeuronClusterOptimizerSyncCallback,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)

CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC = NeuronClusterOptimizerSyncCallback()


## Cluster Geometry Options
CLUSTER_X_AXIS_TOTAL_NEURONS: int = 10
CLUSTER_Y_AXIS_TOTAL_NEURONS: int = 10
CLUSTER_Z_AXIS_TOTAL_NEURONS: int = 1
CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS: int = 3
CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS: int = 3
CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS: int = 1
CLUSTER_MAX_STEPS: int = 4
CLUSTER_GROWTH_THRESHOLD: int | None = 250

## Cluster Terminal Options
CLUSTER_TERMINAL_XY_AXIS_RANGE: TerminalRangeOptions = TerminalRangeOptions.ONE
CLUSTER_TERMINAL_Z_AXIS_RANGE: TerminalRangeOptions = TerminalRangeOptions.ONE
CLUSTER_TERMINAL_Z_AXIS_OFFSET: TerminalZAxisOffsetOptions = (
    TerminalZAxisOffsetOptions.ZERO
)
CLUSTER_TERMINAL_TOP_K: int = 1

### Cluster Terminal Router Options
CLUSTER_TERMINAL_ROUTER_NUM_LAYERS: int = 1
CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM: int = HIDDEN_DIM
CLUSTER_TERMINAL_ROUTER_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
CLUSTER_TERMINAL_ROUTER_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
CLUSTER_TERMINAL_ROUTER_DROPOUT_PROBABILITY: float = 0.0
CLUSTER_TERMINAL_ROUTER_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
CLUSTER_TERMINAL_ROUTER_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
CLUSTER_TERMINAL_ROUTER_BIAS_FLAG: bool = True

### Cluster Terminal Sampler Options
CLUSTER_TERMINAL_SAMPLER_THRESHOLD: float = 0.0
CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES: int = 0
CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

## Cluster Halting Options
CLUSTER_HALTING_FLAG: bool = True
CLUSTER_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
CLUSTER_HALTING_THRESHOLD: float = 0.95
CLUSTER_HALTING_DROPOUT: float = 0.0
CLUSTER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
CLUSTER_HALTING_STACK_HIDDEN_DIM: int = HIDDEN_DIM
CLUSTER_HALTING_OUTPUT_DIM: int = 2
CLUSTER_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
CLUSTER_HALTING_STACK_NUM_LAYERS: int = 1
CLUSTER_HALTING_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY: float = 0.0
CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
CLUSTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
CLUSTER_HALTING_STACK_BIAS_FLAG: bool = True
