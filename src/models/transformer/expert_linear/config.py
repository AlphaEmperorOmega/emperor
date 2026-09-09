from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextSinusoidalPositionalEmbeddingConfig,
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
BATCH_SIZE = 64
LEARNING_RATE = 1.0
VOCAB_SIZE = 8192
MODEL_DIM = 128
RUNTIME_VALUE_CONSTRAINTS = positive_runtime_fields("MODEL_DIM")
SOURCE_SEQUENCE_LENGTH = 64
TARGET_SEQUENCE_LENGTH = 64
SEQUENCE_LENGTH = 64
DROPOUT_PROBABILITY = 0.1
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    TextSinusoidalPositionalEmbeddingConfig
)

# Attention Options
ATTN_NUM_HEADS: int = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False
ATTN_ZERO_ATTENTION_FLAG: bool = False

## Attention Projection Stack Options
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = True
ATTN_STACK_HIDDEN_DIM: int = MODEL_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
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
ATTN_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = False

### Attention Projection Gate Options
ATTN_STACK_GATE_FLAG: bool = False
ATTN_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#### Attention Projection Gate Stack Options
ATTN_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = True
ATTN_GATE_STACK_BIAS_FLAG: bool | None = True

### Attention Projection Halting Options
ATTN_STACK_HALTING_FLAG: bool = False
ATTN_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_HALTING_THRESHOLD: float = 0.999
ATTN_HALTING_DROPOUT: float = 0.0
ATTN_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

#### Attention Projection Halting Stack Options
ATTN_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ATTN_HALTING_STACK_HIDDEN_DIM: int | None = None
ATTN_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
ATTN_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
ATTN_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
ATTN_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
ATTN_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
ATTN_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
ATTN_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
ATTN_HALTING_STACK_BIAS_FLAG: bool | None = None

### Attention Projection Memory Options
ATTN_MEMORY_FLAG: bool = False
ATTN_MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
ATTN_MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
ATTN_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
ATTN_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None

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

### Attention Projection Recurrent Layer Options
ATTN_RECURRENT_FLAG: bool = False
ATTN_RECURRENT_MAX_STEPS: int = 2
ATTN_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_RECURRENT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM

#### Attention Projection Recurrent Gate Options
ATTN_RECURRENT_STACK_GATE_FLAG: bool = False
ATTN_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

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

#### Attention Projection Recurrent Halting Options
ATTN_RECURRENT_STACK_HALTING_FLAG: bool = False
ATTN_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
ATTN_RECURRENT_HALTING_THRESHOLD: float = 0.999
ATTN_RECURRENT_HALTING_DROPOUT: float = 0.0
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
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


# Feed-Forward Stack Options
FF_NUM_LAYERS: int = 1
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = 512
FF_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
FF_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_STACK_DROPOUT_PROBABILITY: float = DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
FF_STACK_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool = False

## Feed-Forward Gate Options
FF_STACK_GATE_FLAG: bool = False
FF_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

### Feed-Forward Gate Stack Options
FF_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_GATE_STACK_HIDDEN_DIM: int | None = None
FF_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_GATE_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_GATE_STACK_NUM_LAYERS: int | None = None
FF_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_GATE_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_GATE_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_GATE_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = True
FF_GATE_STACK_BIAS_FLAG: bool | None = True

## Feed-Forward Halting Options
FF_STACK_HALTING_FLAG: bool = False
FF_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_HALTING_THRESHOLD: float = 0.999
FF_HALTING_DROPOUT: float = 0.0
FF_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)

### Feed-Forward Halting Stack Options
FF_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
FF_HALTING_STACK_NORMALIZATION: NormalizationOptions | None = None
FF_HALTING_STACK_NUM_LAYERS: int | None = None
FF_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
FF_HALTING_STACK_RESIDUAL_MODEL_FLAG: bool = False
FF_HALTING_STACK_RESIDUAL_BLOCK_SIZE: int | None = None
FF_HALTING_STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
FF_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
FF_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG: bool | None = None
FF_HALTING_STACK_BIAS_FLAG: bool | None = None

## Feed-Forward Memory Options
FF_MEMORY_FLAG: bool = False
FF_MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
FF_MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
FF_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
FF_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None

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

## Feed-Forward Recurrent Layer Options
FF_RECURRENT_FLAG: bool = False
FF_RECURRENT_MAX_STEPS: int = 2
FF_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
FF_RECURRENT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM

### Feed-Forward Recurrent Gate Options
FF_RECURRENT_STACK_GATE_FLAG: bool = False
FF_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

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

### Feed-Forward Recurrent Halting Options
FF_RECURRENT_STACK_HALTING_FLAG: bool = False
FF_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
FF_RECURRENT_HALTING_THRESHOLD: float = 0.999
FF_RECURRENT_HALTING_DROPOUT: float = 0.0
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
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

# Transformer Options
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
ENCODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
ENCODER_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
ENCODER_OUTPUT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM
DECODER_OUTPUT_NORMALIZATION: NormalizationOptions = NormalizationOptions.LAYER_NORM
DECODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
DECODER_NORMALIZATION: NormalizationOptions = NormalizationOptions.RMS_NORM
ENCODER_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_SELF_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_CROSS_ATTN_NUM_HEADS = ATTN_NUM_HEADS
ENCODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
DECODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
ENCODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS
DECODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS

# Mixture Of Experts Options
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: bool | None = None
NUM_EXPERTS = 4
TOP_K = 2
DROPPED_TOKEN_BEHAVIOR: DroppedTokenOptions = DroppedTokenOptions.ZEROS
COMPUTE_EXPERT_MIXTURE_FLAG: bool = True
WEIGHTED_PARAMETERS_FLAG: bool = False
WEIGHTING_POSITION_OPTION: ExpertWeightingPositionOptions = (
    ExpertWeightingPositionOptions.BEFORE_EXPERTS
)
ROUTING_INITIALIZATION_MODE: RoutingInitializationMode = RoutingInitializationMode.LAYER
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
NORMALIZE_PROBABILITIES_FLAG = True
SAMPLER_NOISY_TOPK_FLAG: bool = False
COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SWITCH_LOSS_WEIGHT = 0.0
ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0
CAPACITY_FACTOR = 0.0
ROUTER_NOISY_TOPK_FLAG: bool = False

# Controller Options
STACK_GATE_FLAG = False
STACK_HALTING_FLAG = False
HALTING_THRESHOLD: float | None = None
MEMORY_FLAG = False
RECURRENT_FLAG = False
RECURRENT_STACK_GATE_FLAG = False
RECURRENT_STACK_HALTING_FLAG = False
RECURRENT_HALTING_THRESHOLD: float | None = None
RECURRENT_MAX_STEPS = 2
RECURRENT_INITIAL_ITERATIONS: int = 2
RECURRENT_GRADIENT_TRANSITION_COUNT: int | None = None
RECURRENT_NO_GRADIENT_TRANSITION_COUNT: int | None = None
RECURRENT_ITERATION_INCREMENT: int = 1
RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT: int = 1
RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG: bool = False
RECURRENT_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
RECURRENT_RESIDUAL_MODEL_FLAG: bool = False
RECURRENT_RESIDUAL_BLOCK_SIZE: int | None = None
RECURRENT_RESIDUAL_RMS_NORM_EPSILON: float | None = None

## Residual Options
# - False uses the residual variant's learned query or coefficient parameters.
# - True uses the residual stack for input-dependent queries or coefficients.
# - For every attention residual selector, supply its RESIDUAL_BLOCK_SIZE and
#   RESIDUAL_RMS_NORM_EPSILON explicitly (suggested: 1 for full attention, 1e-6).
STACK_RESIDUAL_CONNECTION_OPTION: type[ResidualConfig] | None = None
STACK_RESIDUAL_MODEL_FLAG: bool = False
STACK_RESIDUAL_BLOCK_SIZE: int | None = None
STACK_RESIDUAL_RMS_NORM_EPSILON: float | None = None
### Residual Stack Options
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


# Trainer
NUM_EPOCHS = 30
TRAINER_ACCELERATOR: str = "auto"
TRAINER_DEVICES: str | int = "auto"
TRAINER_ACCUMULATE_GRAD_BATCHES: int = 1
TRAINER_PRECISION: str = "32-true"
TRAINER_DETERMINISTIC = True
TRAINER_BENCHMARK: bool = False
TRAINER_MAX_STEPS: int = -1
TRAINER_MAX_TIME: str | None = None
TRAINER_VAL_CHECK_INTERVAL: float = 1.0
TRAINER_LIMIT_TRAIN_BATCHES: float = 1.0
TRAINER_LIMIT_VAL_BATCHES: float = 1.0
TRAINER_OVERFIT_BATCHES: int | float = 0.0
TRAINER_NUM_SANITY_VAL_STEPS: int = 2
TRAINER_GRADIENT_CLIP_VAL = 1.0
TRAINER_GRADIENT_CLIP_ALGORITHM = "norm"
TRAINER_LOG_EVERY_N_STEPS = 10
TRAINER_ENABLE_PROGRESS_BAR: bool = False
TRAINER_ENABLE_CHECKPOINTING: bool = False
TRAINER_ENABLE_MODEL_SUMMARY: bool = False
TRAINER_PROFILER: str | None = None

MONITOR_LOG_EVERY_N_STEPS: int = 100

# Callback
CALLBACK_EARLY_STOPPING_PATIENCE = 5
CALLBACK_EARLY_STOPPING_METRIC = "validation/loss"
CALLBACK_EARLY_STOPPING_MIN_DELTA = 0.0
CALLBACK_EARLY_STOPPING_STRICT = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE = True
CALLBACK_CHECKPOINT_FLAG = False

# Run
DATA_NUM_WORKERS = 0
RUN_TEST_AFTER_FIT = True
SEED: int | None = 0


_CONFIG_FIELD_METADATA_ALIASES: dict[str, str] = {}


def _copy_path_runtime_defaults(source_prefix: str, target_prefix: str) -> None:
    annotations = globals().setdefault("__annotations__", {})
    for name, value in tuple(globals().items()):
        if not name.isupper() or not name.startswith(source_prefix):
            continue
        target = f"{target_prefix}{name.removeprefix(source_prefix)}"
        globals()[target] = value
        _CONFIG_FIELD_METADATA_ALIASES[target] = name
        if name in annotations:
            annotations[target] = annotations[name]


for _target_prefix in (
    "ENCODER_ATTN_",
    "DECODER_SELF_ATTN_",
    "DECODER_CROSS_ATTN_",
):
    _copy_path_runtime_defaults("ATTN_", _target_prefix)
for _target_prefix in ("ENCODER_FF_", "DECODER_FF_"):
    _copy_path_runtime_defaults("FF_", _target_prefix)
for _target_prefix in ("ROUTER_", "EXPERT_"):
    _copy_path_runtime_defaults("FF_", _target_prefix)

ENCODER_FF_STACK_HIDDEN_DIM = ENCODER_FEED_FORWARD_HIDDEN_DIM
DECODER_FF_STACK_HIDDEN_DIM = DECODER_FEED_FORWARD_HIDDEN_DIM
ENCODER_FF_NUM_LAYERS = ENCODER_FEED_FORWARD_NUM_LAYERS
DECODER_FF_NUM_LAYERS = DECODER_FEED_FORWARD_NUM_LAYERS
ROUTER_STACK_HIDDEN_DIM = MODEL_DIM
ROUTER_STACK_NUM_LAYERS = 2
ROUTER_STACK_ACTIVATION = ActivationOptions.GELU
EXPERT_STACK_HIDDEN_DIM = MODEL_DIM
EXPERT_STACK_NUM_LAYERS = 1
EXPERT_STACK_ACTIVATION = ActivationOptions.RELU

del _target_prefix
