from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    AdditiveDynamicBiasConfig,  # noqa: F401
    AffineTransformDynamicBiasConfig,  # noqa: F401
    AntiDynamicDiagonalConfig,  # noqa: F401
    AttentionGroupingConfig,  # noqa: F401
    AxisMaskConfig,
    BankExpansionFactorOptions,
    CombinedDynamicDiagonalConfig,  # noqa: F401
    DiagonalAxisMaskConfig,  # noqa: F401
    DiagonallyModulatedLowRankDynamicWeightConfig,  # noqa: F401
    DualModelDynamicWeightConfig,  # noqa: F401
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,  # noqa: F401
    GroupingConfig,
    HypernetworkDynamicWeightConfig,  # noqa: F401
    LayeredWeightedBankDynamicWeightConfig,  # noqa: F401
    LowRankDynamicWeightConfig,  # noqa: F401
    LowRankFactorSourceOptions,
    MaskDimensionOptions,
    MatrixBiasMixtureConfig,  # noqa: F401
    MatrixWeightsMixtureConfig,  # noqa: F401
    MeanGroupingConfig,  # noqa: F401
    MeanStdGroupingConfig,  # noqa: F401
    MultiplicativeDynamicBiasConfig,  # noqa: F401
    OuterProductMaskConfig,  # noqa: F401
    PerAxisScoreMaskConfig,  # noqa: F401
    RMSGroupingConfig,  # noqa: F401
    SigmoidGatedDynamicBiasConfig,  # noqa: F401
    SingleModelDynamicWeightConfig,  # noqa: F401
    SoftWeightedBankDynamicWeightConfig,  # noqa: F401
    StandardDynamicDiagonalConfig,  # noqa: F401
    SumGroupingConfig,  # noqa: F401
    SummaryNormalizationOptions,
    TanhGatedDynamicBiasConfig,  # noqa: F401
    TopSliceAxisMaskConfig,  # noqa: F401
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,  # noqa: F401
    WeightInformedScoreAxisMaskConfig,  # noqa: F401
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.config import ConfigBase
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    ImageLearnedPositionalEmbeddingConfig,
)
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,  # noqa: F401
    AttentionResidualConfig,  # noqa: F401
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
    WeightedBlendResidualConfig,  # noqa: F401
    WeightedResidualConfig,  # noqa: F401
)
from emperor.memory import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from model_runtime.packages.runtime_values import positive_runtime_fields

# Global
INPUT_DIM: int = 224 * 224 * 3
HIDDEN_DIM: int = 32
RUNTIME_VALUE_CONSTRAINTS = positive_runtime_fields("HIDDEN_DIM")
OUTPUT_DIM: int = 10
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
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
SEED: int | None = None

# Callback
CALLBACK_EARLY_STOPPING_PATIENCE: int = 0
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"
CALLBACK_EARLY_STOPPING_MIN_DELTA: float = 0.0
CALLBACK_EARLY_STOPPING_STRICT: bool = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: bool = True
CALLBACK_CHECKPOINT_FLAG: bool = False

#########################################################################
# IMAGE PATCHES
IMAGE_PATCH_SIZE: int = 16
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 224
PATCH_DROPOUT_PROBABILITY: float = 0.0
PATCH_BIAS_FLAG: bool = True

#########################################################################
# POSITIONAL EMBEDDING (added to patch embeddings before the encoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    ImageLearnedPositionalEmbeddingConfig
)
# Padding index note: use None, not 0. Images have no padding token, so the
# [CLS] slot at position 0 must learn its positional embedding rather than
# being zero-frozen by an nn.Embedding padding_idx. See proper_vit.md.
POSITIONAL_EMBEDDING_PADDING_IDX: int | None = None
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool = False

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: int = 12
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.0
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
ENCODER_OUTPUT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = True
STACK_BIAS_FLAG: bool = True

#########################################################################
# Layer Stack Submodule Options
SUBMODULE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
SUBMODULE_STACK_NORMALIZATION: NormalizationOptions = NORMALIZATION
SUBMODULE_STACK_NUM_LAYERS: int = 2
SUBMODULE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
SUBMODULE_STACK_RESIDUAL_MODEL_FLAG: bool = False
SUBMODULE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
SUBMODULE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
SUBMODULE_STACK_DROPOUT_PROBABILITY: float = 0.0
SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = False
SUBMODULE_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# Residual Options
# - False uses the residual variant's learned query or coefficient parameters.
# - True uses the residual stack for input-dependent queries or coefficients.
# - For every attention residual selector, supply its RESIDUAL_BLOCK_SIZE and
#   RESIDUAL_RMS_NORM_EPSILON explicitly (suggested: 1 for full attention, 1e-6).
STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
STACK_RESIDUAL_MODEL_FLAG: bool = False
STACK_RESIDUAL_BLOCK_SIZE: int | None = None
STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
## Residual Stack Options
# - If False, residual stack options inherit layer stack submodule options.
RESIDUAL_STACK_INDEPENDENT_FLAG: bool = False
RESIDUAL_STACK_HIDDEN_DIM: int | None = None
RESIDUAL_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RESIDUAL_STACK_NORMALIZATION: NormalizationOptions | None = None
RESIDUAL_STACK_NUM_LAYERS: int | None = None
RESIDUAL_STACK_ACTIVATION: ActivationOptions | None = None
RESIDUAL_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
RESIDUAL_STACK_RESIDUAL_MODEL_FLAG: bool = False
RESIDUAL_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
RESIDUAL_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
RESIDUAL_STACK_DROPOUT_PROBABILITY: float | None = None
RESIDUAL_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RESIDUAL_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
RESIDUAL_STACK_BIAS_FLAG: bool | None = None


#########################################################################
# Gate Options
STACK_GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Gate Stack Options
GATE_STACK_INDEPENDENT_FLAG: bool = False
GATE_STACK_HIDDEN_DIM: int | None = None
GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
GATE_STACK_NUM_LAYERS: int | None = None
GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
GATE_STACK_DROPOUT_PROBABILITY: float | None = None
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = True
GATE_STACK_BIAS_FLAG: bool | None = True

#########################################################################
# Halting Options
STACK_HALTING_FLAG: bool = False
HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
HALTING_THRESHOLD: float = 0.999
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_OUTPUT_DIM: int = 2
## Halting Stack Options
HALTING_STACK_INDEPENDENT_FLAG: bool = False
HALTING_STACK_HIDDEN_DIM: int | None = None
HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
HALTING_STACK_NUM_LAYERS: int | None = None
HALTING_STACK_ACTIVATION: ActivationOptions | None = None
HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Memory Options
MEMORY_FLAG: bool = False
MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Memory Stack Options
MEMORY_STACK_INDEPENDENT_FLAG: bool = False
MEMORY_STACK_HIDDEN_DIM: int | None = None
MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
MEMORY_STACK_NORMALIZATION: NormalizationOptions | None = None
MEMORY_STACK_NUM_LAYERS: int | None = None
MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
MEMORY_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Recurrent Layer Options
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_INITIAL_ITERATIONS: int = 2
RECURRENT_GRADIENT_TRANSITION_COUNT: int | None = None
RECURRENT_NO_GRADIENT_TRANSITION_COUNT: int | None = None
RECURRENT_ITERATION_INCREMENT: int = 1
RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT: int = 1
RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG: bool = False
RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
RECURRENT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM

## Recurrent Gate Options
RECURRENT_STACK_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
### Recurrent Gate Stack Options
RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RECURRENT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

## Recurrent Halting Options
RECURRENT_STACK_HALTING_FLAG: bool = False
RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
RECURRENT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
RECURRENT_HALTING_DROPOUT: float = HALTING_DROPOUT
RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
### Recurrent Halting Stack Options
RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
RECURRENT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None


#########################################################################
# Attention Options
ATTN_NUM_HEADS: int = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False

## Attention Projection Stack Options
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = True
ATTN_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = True

#########################################################################
### Attention Projection Gate Options
ATTN_STACK_GATE_FLAG: bool = False
ATTN_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
ATTN_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
#### Attention Projection Gate Stack Options
ATTN_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
ATTN_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
### Attention Projection Halting Options
ATTN_STACK_HALTING_FLAG: bool = False
ATTN_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_HALTING_THRESHOLD: float = HALTING_THRESHOLD
ATTN_HALTING_DROPOUT: float = HALTING_DROPOUT
ATTN_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
#### Attention Projection Halting Stack Options
ATTN_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ATTN_HALTING_STACK_HIDDEN_DIM: int | None = None
ATTN_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    HALTING_STACK_LAYER_NORM_POSITION
)
ATTN_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = (
    HALTING_STACK_NORMALIZATION
)
ATTN_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ATTN_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Attention Projection Memory Options
ATTN_MEMORY_FLAG: bool = False
ATTN_MEMORY_OPTION: type[DynamicMemoryConfig] = MEMORY_OPTION
ATTN_MEMORY_POSITION_OPTION: MemoryPositionOptions = MEMORY_POSITION_OPTION
ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = (
    MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
)
ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = (
    MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
)
#### Attention Projection Memory Stack Options
ATTN_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
ATTN_MEMORY_STACK_HIDDEN_DIM: int | None = None
ATTN_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_MEMORY_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_MEMORY_STACK_NUM_LAYERS: int | None = None
ATTN_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ATTN_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Attention Projection Recurrent Layer Options
ATTN_RECURRENT_FLAG: bool = False
ATTN_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
ATTN_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)
ATTN_RECURRENT_NORMALIZATION: NormalizationOptions = RECURRENT_NORMALIZATION

#########################################################################
#### Attention Projection Recurrent Gate Options
ATTN_RECURRENT_STACK_GATE_FLAG: bool = False
ATTN_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
ATTN_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
##### Attention Projection Recurrent Gate Stack Options
ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_RECURRENT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ATTN_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
#### Attention Projection Recurrent Halting Options
ATTN_RECURRENT_STACK_HALTING_FLAG: bool = False
ATTN_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
ATTN_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
)
##### Attention Projection Recurrent Halting Stack Options
ATTN_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ATTN_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
ATTN_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_RECURRENT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Feed-Forward Stack Options
FF_NUM_LAYERS: int = 2
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = HIDDEN_DIM
FF_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
FF_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
FF_STACK_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = True

#########################################################################
## Feed-Forward Gate Options
FF_STACK_GATE_FLAG: bool = False
FF_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
FF_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
### Feed-Forward Gate Stack Options
FF_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_GATE_STACK_HIDDEN_DIM: int | None = None
FF_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_GATE_STACK_NUM_LAYERS: int | None = None
FF_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
FF_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
## Feed-Forward Halting Options
FF_STACK_HALTING_FLAG: bool = False
FF_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_HALTING_THRESHOLD: float = HALTING_THRESHOLD
FF_HALTING_DROPOUT: float = HALTING_DROPOUT
FF_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = HALTING_HIDDEN_STATE_MODE
### Feed-Forward Halting Stack Options
FF_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    HALTING_STACK_LAYER_NORM_POSITION
)
FF_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = (
    HALTING_STACK_NORMALIZATION
)
FF_HALTING_STACK_NUM_LAYERS: int | None = None
FF_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
FF_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
FF_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Feed-Forward Memory Options
FF_MEMORY_FLAG: bool = False
FF_MEMORY_OPTION: type[DynamicMemoryConfig] = MEMORY_OPTION
FF_MEMORY_POSITION_OPTION: MemoryPositionOptions = MEMORY_POSITION_OPTION
FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = (
    MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
)
FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = (
    MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
)
### Feed-Forward Memory Stack Options
FF_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
FF_MEMORY_STACK_HIDDEN_DIM: int | None = None
FF_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_MEMORY_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_MEMORY_STACK_NUM_LAYERS: int | None = None
FF_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
FF_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Feed-Forward Recurrent Layer Options
FF_RECURRENT_FLAG: bool = False
FF_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
FF_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)
FF_RECURRENT_NORMALIZATION: NormalizationOptions = RECURRENT_NORMALIZATION

#########################################################################
### Feed-Forward Recurrent Gate Options
FF_RECURRENT_STACK_GATE_FLAG: bool = False
FF_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
FF_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
#### Feed-Forward Recurrent Gate Stack Options
FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_RECURRENT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
FF_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Feed-Forward Recurrent Halting Options
FF_RECURRENT_STACK_HALTING_FLAG: bool = False
FF_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
FF_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
)
#### Feed-Forward Recurrent Halting Stack Options
FF_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_RECURRENT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
FF_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# OUTPUT HEAD (maps [CLS] hidden_dim to class logits)
OUTPUT_BIAS_FLAG: bool = True

#########################################################################
# Mixture Of Experts Model Options
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: bool = False
TOP_K: int = 3
NUM_EXPERTS: int = 12
CAPACITY_FACTOR: float = 0.0
DROPPED_TOKEN_BEHAVIOR: DroppedTokenOptions = DroppedTokenOptions.ZEROS
COMPUTE_EXPERT_MIXTURE_FLAG: bool = True
WEIGHTED_PARAMETERS_FLAG: bool = True
WEIGHTING_POSITION_OPTION: ExpertWeightingPositionOptions = (
    ExpertWeightingPositionOptions.AFTER_EXPERTS
)
ROUTING_INITIALIZATION_MODE: RoutingInitializationMode = RoutingInitializationMode.LAYER

#########################################################################
## Expert Stack Options
EXPERT_STACK_HIDDEN_DIM: int = SUBMODULE_STACK_HIDDEN_DIM
EXPERT_STACK_NUM_LAYERS: int = SUBMODULE_STACK_NUM_LAYERS
EXPERT_STACK_ACTIVATION: ActivationOptions = SUBMODULE_STACK_ACTIVATION
EXPERT_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
)
EXPERT_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_STACK_DROPOUT_PROBABILITY: float = SUBMODULE_STACK_DROPOUT_PROBABILITY
EXPERT_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_STACK_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
EXPERT_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = True
EXPERT_BIAS_FLAG: bool = SUBMODULE_STACK_BIAS_FLAG

#########################################################################
## Expert Gate Options
# If `EXPERT_STACK_GATE_FLAG` is False, the expert gate parameters below are ignored.
EXPERT_STACK_GATE_FLAG: bool = False
EXPERT_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
EXPERT_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
### Expert Gate Stack Options
# If False, expert gate stack options inherit the expert stack options.
EXPERT_GATE_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_GATE_STACK_HIDDEN_DIM: int | None = None
EXPERT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
EXPERT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
EXPERT_GATE_STACK_NUM_LAYERS: int | None = None
EXPERT_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
EXPERT_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
## Expert Halting Options
# If `EXPERT_STACK_HALTING_FLAG` is False, the expert halting parameters are ignored.
EXPERT_STACK_HALTING_FLAG: bool = False
EXPERT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
EXPERT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
EXPERT_HALTING_DROPOUT: float = HALTING_DROPOUT
EXPERT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
EXPERT_HALTING_OUTPUT_DIM: int = 2
### Expert Halting Stack Options
# If False, expert halting stack options inherit the expert stack options.
EXPERT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_HALTING_STACK_HIDDEN_DIM: int | None = None
EXPERT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    HALTING_STACK_LAYER_NORM_POSITION
)
EXPERT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = (
    HALTING_STACK_NORMALIZATION
)
EXPERT_HALTING_STACK_NUM_LAYERS: int | None = None
EXPERT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
EXPERT_MEMORY_STACK_NORMALIZATION: NormalizationOptions | None = None
EXPERT_MEMORY_STACK_NUM_LAYERS: int | None = None
EXPERT_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
EXPERT_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Expert Recurrent Layer Options
# If `EXPERT_RECURRENT_FLAG` is False, expert recurrence is disabled.
EXPERT_RECURRENT_FLAG: bool = False
EXPERT_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
EXPERT_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)
EXPERT_RECURRENT_NORMALIZATION: NormalizationOptions = RECURRENT_NORMALIZATION

#########################################################################
### Expert Recurrent Gate Options
EXPERT_RECURRENT_STACK_GATE_FLAG: bool = False
EXPERT_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
EXPERT_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
#### Expert Recurrent Gate Stack Options
# If False, expert recurrent gate stack options inherit the expert stack options.
EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
EXPERT_RECURRENT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
EXPERT_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Expert Recurrent Halting Options
EXPERT_RECURRENT_STACK_HALTING_FLAG: bool = False
EXPERT_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
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
EXPERT_RECURRENT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
EXPERT_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ROUTER_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_STACK_DROPOUT_PROBABILITY: float = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
ROUTER_STACK_NORMALIZATION: NormalizationOptions = NORMALIZATION
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = False
ROUTER_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
### Router Gate Options
# If `ROUTER_STACK_GATE_FLAG` is False, the router gate parameters are ignored.
ROUTER_STACK_GATE_FLAG: bool = False
ROUTER_GATE_OPTION: LayerGateOptions | None = EXPERT_GATE_OPTION
ROUTER_GATE_ACTIVATION: ActivationOptions | None = EXPERT_GATE_ACTIVATION
#### Router Gate Stack Options
# If False, router gate stack options inherit layer stack submodule options.
ROUTER_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_GATE_STACK_ACTIVATION: ActivationOptions | None = EXPERT_GATE_STACK_ACTIVATION
ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG
)
ROUTER_GATE_STACK_BIAS_FLAG: bool | None = EXPERT_GATE_STACK_BIAS_FLAG

#########################################################################
### Router Halting Options
# If `ROUTER_STACK_HALTING_FLAG` is False, the router halting parameters are ignored.
ROUTER_STACK_HALTING_FLAG: bool = False
ROUTER_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
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
ROUTER_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = (
    EXPERT_HALTING_STACK_NORMALIZATION
)
ROUTER_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
ROUTER_MEMORY_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_MEMORY_STACK_NUM_LAYERS: int | None = None
ROUTER_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_MEMORY_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_MEMORY_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ROUTER_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Router Recurrent Layer Options
# If `ROUTER_RECURRENT_FLAG` is False, router recurrence is disabled.
ROUTER_RECURRENT_FLAG: bool = False
ROUTER_RECURRENT_MAX_STEPS: int = EXPERT_RECURRENT_MAX_STEPS
ROUTER_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    EXPERT_RECURRENT_LAYER_NORM_POSITION
)
ROUTER_RECURRENT_NORMALIZATION: NormalizationOptions = EXPERT_RECURRENT_NORMALIZATION

#########################################################################
#### Router Recurrent Gate Options
ROUTER_RECURRENT_STACK_GATE_FLAG: bool = False
ROUTER_RECURRENT_GATE_OPTION: LayerGateOptions | None = EXPERT_RECURRENT_GATE_OPTION
ROUTER_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = (
    EXPERT_RECURRENT_GATE_ACTIVATION
)
##### Router Recurrent Gate Stack Options
# If False, router recurrent gate stack options inherit layer stack submodule options.
ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_RECURRENT_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
#### Router Recurrent Halting Options
ROUTER_RECURRENT_STACK_HALTING_FLAG: bool = False
ROUTER_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ROUTER_RECURRENT_HALTING_THRESHOLD: float = EXPERT_RECURRENT_HALTING_THRESHOLD
ROUTER_RECURRENT_HALTING_DROPOUT: float = EXPERT_RECURRENT_HALTING_DROPOUT
ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
)
##### Router Recurrent Halting Stack Options
# If False, router recurrent halting stack options inherit layer stack submodule
# options.
ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Adaptive Generator Stack Options
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LAYER_NORM_POSITION
)
ADAPTIVE_GENERATOR_STACK_NORMALIZATION: NormalizationOptions = NORMALIZATION
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ADAPTIVE_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# Weight Generator Options
# If `WEIGHT_OPTION_FLAG` is False, the expert weight parameters below are ignored.
WEIGHT_OPTION_FLAG: bool = False
WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
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
WEIGHT_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
BIAS_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = None
BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
DIAGONAL_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = None
DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
MASK_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
MASK_GENERATOR_STACK_NUM_LAYERS: int | None = None
MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = None


#########################################################################
# Sampler Model Options
## Router Options
### Router Weight Generator Options
# If `ROUTER_WEIGHT_OPTION_FLAG` is False, router weight parameters are ignored.
ROUTER_WEIGHT_OPTION_FLAG: bool = False
ROUTER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
ROUTER_GENERATOR_DEPTH: DynamicDepthOptions = GENERATOR_DEPTH
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
ROUTER_WEIGHT_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
ROUTER_BIAS_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
ROUTER_DIAGONAL_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
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
ROUTER_MASK_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = (
    None
)
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = None


# ==============================================================================
# Main Adaptive Generation
# ==============================================================================
WEIGHT_INPUT_FACTOR_SOURCE: LowRankFactorSourceOptions | None = None
WEIGHT_OUTPUT_FACTOR_SOURCE: LowRankFactorSourceOptions | None = None
WEIGHT_MIXTURE_NUM_EXPERTS: int | None = None
BIAS_MIXTURE_NUM_EXPERTS: int | None = None
WEIGHT_MIXTURE_TOP_K: int | None = None
BIAS_MIXTURE_TOP_K: int | None = None
WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: bool | None = None
BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: bool | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: int | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: bool | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: int | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    None
)
WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: bool | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: int | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: bool | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: int | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: int | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: int | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: int | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = (
    None
)
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: bool | None = None
BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: bool | None = None

# Main Grouping
GROUPING_SCOPE: AdaptiveParameterGroupingScopeOptions | None = None
GROUP_COUNT: int | None = None
CHUNK_SIZE: int | None = None
GROUPING_SEQUENCE_LENGTH: int | None = None
GROUPING_INPUT_ORDER: AdaptiveParameterInputOrderOptions | None = None
GROUPING_METHOD: type[GroupingConfig] | None = None
GROUPING_SUMMARY_NORMALIZATION: SummaryNormalizationOptions | None = None
GROUPING_MODEL_CONFIG: ConfigBase | None = None
GROUPING_ATTENTION_HIDDEN_DIM: int | None = None
GROUPING_RMS_NORM_EPSILON: float | None = None


# ==============================================================================
# Router Adaptive Generation
# ==============================================================================
ROUTER_WEIGHT_INPUT_FACTOR_SOURCE: LowRankFactorSourceOptions | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_SOURCE: LowRankFactorSourceOptions | None = None
ROUTER_WEIGHT_MIXTURE_NUM_EXPERTS: int | None = None
ROUTER_BIAS_MIXTURE_NUM_EXPERTS: int | None = None
ROUTER_WEIGHT_MIXTURE_TOP_K: int | None = None
ROUTER_BIAS_MIXTURE_TOP_K: int | None = None
ROUTER_WEIGHT_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: bool | None = None
ROUTER_BIAS_MIXTURE_NORMALIZE_PROBABILITIES_FLAG: bool | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: (
    NormalizationOptions | None
) = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = (
    None
)
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: (
    bool | None
) = None
ROUTER_WEIGHT_INPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: bool | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NORMALIZATION: (
    NormalizationOptions | None
) = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = (
    None
)
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: (
    bool | None
) = None
ROUTER_WEIGHT_OUTPUT_FACTOR_GENERATOR_STACK_BIAS_FLAG: bool | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NORMALIZATION: NormalizationOptions | None = (
    None
)
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: (
    bool | None
) = None
ROUTER_WEIGHT_COEFFICIENT_GENERATOR_STACK_BIAS_FLAG: bool | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: (
    NormalizationOptions | None
) = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NORMALIZATION: (
    NormalizationOptions | None
) = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    type[ResidualConfig] | None
) = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = (
    None
)
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_MODEL_FLAG: bool = False
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = (
    None
)
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: (
    bool | None
) = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: (
    bool | None
) = None
ROUTER_WEIGHT_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: bool | None = None
ROUTER_BIAS_MIXTURE_ROUTER_GENERATOR_STACK_BIAS_FLAG: bool | None = None

# Router Grouping
ROUTER_GROUPING_SCOPE: AdaptiveParameterGroupingScopeOptions | None = None
ROUTER_GROUP_COUNT: int | None = None
ROUTER_CHUNK_SIZE: int | None = None
ROUTER_GROUPING_SEQUENCE_LENGTH: int | None = None
ROUTER_GROUPING_INPUT_ORDER: AdaptiveParameterInputOrderOptions | None = None
ROUTER_GROUPING_METHOD: type[GroupingConfig] | None = None
ROUTER_GROUPING_SUMMARY_NORMALIZATION: SummaryNormalizationOptions | None = None
ROUTER_GROUPING_MODEL_CONFIG: ConfigBase | None = None
ROUTER_GROUPING_ATTENTION_HIDDEN_DIM: int | None = None
ROUTER_GROUPING_RMS_NORM_EPSILON: float | None = None


# Attn Grouping
ATTN_GROUPING_SCOPE: AdaptiveParameterGroupingScopeOptions | None = None
ATTN_GROUP_COUNT: int | None = None
ATTN_CHUNK_SIZE: int | None = None
ATTN_GROUPING_SEQUENCE_LENGTH: int | None = None
ATTN_GROUPING_INPUT_ORDER: AdaptiveParameterInputOrderOptions | None = None
ATTN_GROUPING_METHOD: type[GroupingConfig] | None = None
ATTN_GROUPING_SUMMARY_NORMALIZATION: SummaryNormalizationOptions | None = None
ATTN_GROUPING_MODEL_CONFIG: ConfigBase | None = None
ATTN_GROUPING_ATTENTION_HIDDEN_DIM: int | None = None
ATTN_GROUPING_RMS_NORM_EPSILON: float | None = None


# Ff Grouping
FF_GROUPING_SCOPE: AdaptiveParameterGroupingScopeOptions | None = None
FF_GROUP_COUNT: int | None = None
FF_CHUNK_SIZE: int | None = None
FF_GROUPING_SEQUENCE_LENGTH: int | None = None
FF_GROUPING_INPUT_ORDER: AdaptiveParameterInputOrderOptions | None = None
FF_GROUPING_METHOD: type[GroupingConfig] | None = None
FF_GROUPING_SUMMARY_NORMALIZATION: SummaryNormalizationOptions | None = None
FF_GROUPING_MODEL_CONFIG: ConfigBase | None = None
FF_GROUPING_ATTENTION_HIDDEN_DIM: int | None = None
FF_GROUPING_RMS_NORM_EPSILON: float | None = None


# Workbench Config Schema Boundary
# Construction defaults remain available to flat CLI adapters and presets. The
# grouped builder interface exposes residual and adaptive parameter settings.
_PUBLIC_CONFIG_KEYS = {
    "GENERATOR_DEPTH",
    "ROW_MASK_OPTION",
    "GROUP_COUNT",
    "CHUNK_SIZE",
    "INPUT_DIM",
    "OUTPUT_DIM",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
}
_PUBLIC_CONFIG_PREFIXES = (
    "ADAPTIVE_GENERATOR_STACK_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
    "GROUPING_",
    "ATTN_ADAPTIVE_GENERATOR_STACK_",
    "ATTN_WEIGHT_",
    "ATTN_BIAS_",
    "ATTN_DIAGONAL_",
    "ATTN_MASK_",
    "ATTN_GROUPING_",
    "FF_ADAPTIVE_GENERATOR_STACK_",
    "FF_WEIGHT_",
    "FF_BIAS_",
    "FF_DIAGONAL_",
    "FF_MASK_",
    "FF_GROUPING_",
    "ROUTER_WEIGHT_",
    "ROUTER_BIAS_",
    "ROUTER_GROUPING_",
    "ROUTER_DIAGONAL_",
    "ROUTER_MASK_",
    "TRAINER_",
    "CALLBACK_",
    "DATA_",
    "RUN_",
    "MONITOR_",
    "RESIDUAL_STACK_",
)
_PUBLIC_CONFIG_SUFFIXES = (
    "_GROUP_COUNT",
    "_CHUNK_SIZE",
    "_GENERATOR_DEPTH",
    "_ROW_MASK_OPTION",
    "_RESIDUAL_CONNECTION_OPTION",
    "_RESIDUAL_BLOCK_SIZE",
    "_RESIDUAL_RMS_NORM_EPSILON",
    "_RESIDUAL_MODEL_FLAG",
)
CONFIG_SCHEMA_SKIP_KEYS: set[str] = {
    key
    for key in globals()
    if key.isupper()
    and not key.startswith("_")
    and key != "CONFIG_SCHEMA_SKIP_KEYS"
    and key not in _PUBLIC_CONFIG_KEYS
    and not key.startswith(_PUBLIC_CONFIG_PREFIXES)
    and not key.endswith(_PUBLIC_CONFIG_SUFFIXES)
}
