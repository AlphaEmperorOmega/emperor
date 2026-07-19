from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from emperor.memory.options import MemoryPositionOptions

# Package-local trainer defaults. Model packages intentionally do not import the
# repository-level trainer configuration so they remain independently loadable.
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

# Global
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 10

# Callback
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"

# Model
INPUT_DIM: int = 28**2
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 10

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
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
SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
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
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
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
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_OUTPUT_DIM: int = 2
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
# Mixture Of Experts Model Options
EXPERT_TOP_K: int = 3
EXPERT_NUM_EXPERTS: int = 12
EXPERT_CAPACITY_FACTOR: float = 0.0
EXPERT_DROPPED_TOKEN_BEHAVIOR: DroppedTokenOptions = DroppedTokenOptions.ZEROS
EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG: bool = True
EXPERT_WEIGHTED_PARAMETERS_FLAG: bool = True
EXPERT_WEIGHTING_POSITION_OPTION: ExpertWeightingPositionOptions = (
    ExpertWeightingPositionOptions.AFTER_EXPERTS
)
EXPERT_ROUTING_INITIALIZATION_MODE: RoutingInitializationMode = (
    RoutingInitializationMode.LAYER
)

#########################################################################
## Expert Stack Options
EXPERT_STACK_HIDDEN_DIM: int = SUBMODULE_STACK_HIDDEN_DIM
EXPERT_STACK_NUM_LAYERS: int = SUBMODULE_STACK_NUM_LAYERS
EXPERT_STACK_ACTIVATION: ActivationOptions = SUBMODULE_STACK_ACTIVATION
EXPERT_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
)
EXPERT_STACK_DROPOUT_PROBABILITY: float = SUBMODULE_STACK_DROPOUT_PROBABILITY
EXPERT_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
EXPERT_BIAS_FLAG: bool = SUBMODULE_STACK_BIAS_FLAG

#########################################################################
## Expert Gate Options
# If `EXPERT_GATE_FLAG` is False, the expert gate parameters below are ignored.
EXPERT_GATE_FLAG: bool = False
EXPERT_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
EXPERT_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
### Expert Gate Stack Options
# If False, expert gate stack options inherit the expert stack options.
EXPERT_GATE_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_GATE_STACK_HIDDEN_DIM: int | None = None
EXPERT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
EXPERT_GATE_STACK_NUM_LAYERS: int | None = None
EXPERT_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
EXPERT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
EXPERT_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
## Expert Halting Options
# If `EXPERT_HALTING_FLAG` is False, the expert halting parameters are ignored.
EXPERT_HALTING_FLAG: bool = False
EXPERT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
EXPERT_HALTING_DROPOUT: float = HALTING_DROPOUT
EXPERT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
EXPERT_HALTING_OUTPUT_DIM: int = HALTING_OUTPUT_DIM
### Expert Halting Stack Options
# If False, expert halting stack options inherit the expert stack options.
EXPERT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_HALTING_STACK_HIDDEN_DIM: int | None = None
EXPERT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    HALTING_STACK_LAYER_NORM_POSITION
)
EXPERT_HALTING_STACK_NUM_LAYERS: int | None = None
EXPERT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
EXPERT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
EXPERT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
EXPERT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Expert Memory Options
# If `EXPERT_MEMORY_FLAG` is False, the expert memory parameters are ignored.
EXPERT_MEMORY_FLAG: bool = False
EXPERT_MEMORY_OPTION: type[DynamicMemoryConfig] = MEMORY_OPTION
EXPERT_MEMORY_POSITION_OPTION: MemoryPositionOptions = MEMORY_POSITION_OPTION
EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = (
    MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
)
EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = (
    MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
)
### Expert Memory Stack Options
# If False, expert memory stack options inherit the expert stack options.
EXPERT_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_MEMORY_STACK_HIDDEN_DIM: int | None = None
EXPERT_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
EXPERT_MEMORY_STACK_NUM_LAYERS: int | None = None
EXPERT_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
EXPERT_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Expert Recurrent Layer Options
# If `EXPERT_RECURRENT_FLAG` is False, expert recurrence is disabled.
EXPERT_RECURRENT_FLAG: bool = False
EXPERT_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
EXPERT_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
### Expert Recurrent Gate Options
EXPERT_RECURRENT_GATE_FLAG: bool = False
EXPERT_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
EXPERT_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
#### Expert Recurrent Gate Stack Options
# If False, expert recurrent gate stack options inherit the expert stack options.
EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
EXPERT_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Expert Recurrent Halting Options
EXPERT_RECURRENT_HALTING_FLAG: bool = False
EXPERT_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
EXPERT_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
)
#### Expert Recurrent Halting Stack Options
# If False, expert recurrent halting stack options inherit the expert stack options.
EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
EXPERT_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Sampler Model Options
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = True
SAMPLER_NOISY_TOPK_FLAG: bool = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.1
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

#########################################################################
## Router Options
ROUTER_NOISY_TOPK_FLAG: bool = False

#########################################################################
### Router Stack Options
ROUTER_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ROUTER_STACK_NUM_LAYERS: int = 2
ROUTER_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
ROUTER_STACK_DROPOUT_PROBABILITY: float = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = STACK_LAYER_NORM_POSITION
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ROUTER_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
### Router Gate Options
# If `ROUTER_GATE_FLAG` is False, the router gate parameters are ignored.
ROUTER_GATE_FLAG: bool = False
ROUTER_GATE_OPTION: LayerGateOptions | None = EXPERT_GATE_OPTION
ROUTER_GATE_ACTIVATION: ActivationOptions | None = EXPERT_GATE_ACTIVATION
#### Router Gate Stack Options
# If False, router gate stack options inherit layer stack submodule options.
ROUTER_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_GATE_STACK_ACTIVATION: ActivationOptions | None = EXPERT_GATE_STACK_ACTIVATION
ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ROUTER_GATE_STACK_BIAS_FLAG: bool | None = EXPERT_GATE_STACK_BIAS_FLAG

#########################################################################
### Router Halting Options
# If `ROUTER_HALTING_FLAG` is False, the router halting parameters are ignored.
ROUTER_HALTING_FLAG: bool = False
ROUTER_HALTING_THRESHOLD: float = EXPERT_HALTING_THRESHOLD
ROUTER_HALTING_DROPOUT: float = EXPERT_HALTING_DROPOUT
ROUTER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    EXPERT_HALTING_HIDDEN_STATE_MODE
)
ROUTER_HALTING_OUTPUT_DIM: int = EXPERT_HALTING_OUTPUT_DIM
#### Router Halting Stack Options
# If False, router halting stack options inherit layer stack submodule options.
ROUTER_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_HALTING_STACK_HIDDEN_DIM: int | None = None
ROUTER_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    EXPERT_HALTING_STACK_LAYER_NORM_POSITION
)
ROUTER_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
ROUTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Memory Options
# If `ROUTER_MEMORY_FLAG` is False, the router memory parameters are ignored.
ROUTER_MEMORY_FLAG: bool = False
ROUTER_MEMORY_OPTION: type[DynamicMemoryConfig] = EXPERT_MEMORY_OPTION
ROUTER_MEMORY_POSITION_OPTION: MemoryPositionOptions = EXPERT_MEMORY_POSITION_OPTION
ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = (
    EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
)
ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = (
    EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
)
#### Router Memory Stack Options
# If False, router memory stack options inherit layer stack submodule options.
ROUTER_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_MEMORY_STACK_HIDDEN_DIM: int | None = None
ROUTER_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_MEMORY_STACK_NUM_LAYERS: int | None = None
ROUTER_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Recurrent Layer Options
# If `ROUTER_RECURRENT_FLAG` is False, router recurrence is disabled.
ROUTER_RECURRENT_FLAG: bool = False
ROUTER_RECURRENT_MAX_STEPS: int = EXPERT_RECURRENT_MAX_STEPS
ROUTER_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    EXPERT_RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
#### Router Recurrent Gate Options
ROUTER_RECURRENT_GATE_FLAG: bool = False
ROUTER_RECURRENT_GATE_OPTION: LayerGateOptions | None = EXPERT_RECURRENT_GATE_OPTION
ROUTER_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = (
    EXPERT_RECURRENT_GATE_ACTIVATION
)
##### Router Recurrent Gate Stack Options
# If False, router recurrent gate stack options inherit layer stack submodule options.
ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
#### Router Recurrent Halting Options
ROUTER_RECURRENT_HALTING_FLAG: bool = False
ROUTER_RECURRENT_HALTING_THRESHOLD: float = EXPERT_RECURRENT_HALTING_THRESHOLD
ROUTER_RECURRENT_HALTING_DROPOUT: float = EXPERT_RECURRENT_HALTING_DROPOUT
ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
)
##### Router Recurrent Halting Stack Options
# If False, router recurrent halting stack options inherit layer stack submodule options.
ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Weight Generator Options
# If `WEIGHT_OPTION_FLAG` is False, the expert weight parameters below are ignored.
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
# If `BIAS_OPTION_FLAG` is False, the expert bias parameters below are ignored.
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
# If `DIAGONAL_OPTION_FLAG` is False, the expert diagonal parameters below are
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
# If `MASK_OPTION_FLAG` is False, the expert mask parameters below are ignored.
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
OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = WEIGHT_NORMALIZATION_POSITION_OPTION  # noqa: E501
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
# Sampler Model Options
## Router Options
### Router Weight Generator Options
# If `ROUTER_WEIGHT_OPTION_FLAG` is False, router weight parameters are ignored.
ROUTER_WEIGHT_OPTION_FLAG: bool = False
ROUTER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
ROUTER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
ROUTER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
ROUTER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
ROUTER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
ROUTER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
ROUTER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WEIGHT_NORMALIZATION_POSITION_OPTION
)
ROUTER_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
#### Router Weight Generator Stack Options
# If False, router weight generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Bias Generator Options
# If `ROUTER_BIAS_OPTION_FLAG` is False, router bias parameters are ignored.
ROUTER_BIAS_OPTION_FLAG: bool = False
ROUTER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
ROUTER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
ROUTER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
ROUTER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
ROUTER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
#### Router Bias Generator Stack Options
# If False, router bias generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Diagonal Generator Options
# If `ROUTER_DIAGONAL_OPTION_FLAG` is False, router diagonal parameters are ignored.
ROUTER_DIAGONAL_OPTION_FLAG: bool = False
ROUTER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
#### Router Diagonal Generator Stack Options
# If False, router diagonal generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Mask Options
# If `ROUTER_MASK_OPTION_FLAG` is False, router mask parameters are ignored.
ROUTER_MASK_OPTION_FLAG: bool = False
ROUTER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
ROUTER_MASK_THRESHOLD: float = MASK_THRESHOLD
ROUTER_MASK_FLOOR: float = MASK_FLOOR
ROUTER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
ROUTER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
ROUTER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION
#### Router Mask Stack Options
# If False, router mask generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = None
