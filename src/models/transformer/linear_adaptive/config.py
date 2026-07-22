from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,  # noqa: F401
    AffineTransformDynamicBiasConfig,  # noqa: F401
    AntiDynamicDiagonalConfig,  # noqa: F401
    AxisMaskConfig,
    CombinedDynamicDiagonalConfig,  # noqa: F401
    DiagonalAxisMaskConfig,  # noqa: F401
    DualModelDynamicWeightConfig,  # noqa: F401
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    GeneratorDynamicBiasConfig,  # noqa: F401
    HypernetworkDynamicWeightConfig,  # noqa: F401
    LayeredWeightedBankDynamicWeightConfig,  # noqa: F401
    LowRankDynamicWeightConfig,  # noqa: F401
    MultiplicativeDynamicBiasConfig,  # noqa: F401
    OuterProductMaskConfig,  # noqa: F401
    PerAxisScoreMaskConfig,  # noqa: F401
    SigmoidGatedDynamicBiasConfig,  # noqa: F401
    SingleModelDynamicWeightConfig,  # noqa: F401
    SoftWeightedBankDynamicWeightConfig,  # noqa: F401
    StandardDynamicDiagonalConfig,  # noqa: F401
    TanhGatedDynamicBiasConfig,  # noqa: F401
    TopSliceAxisMaskConfig,  # noqa: F401
    WeightedBankDynamicBiasConfig,  # noqa: F401
    WeightInformedScoreAxisMaskConfig,  # noqa: F401
)
from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
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
    ResidualConnectionOptions,
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
BATCH_SIZE = 64
LEARNING_RATE = 1.0
VOCAB_SIZE = 8192
MODEL_DIM = 128
SOURCE_SEQUENCE_LENGTH = 64
TARGET_SEQUENCE_LENGTH = 64
SEQUENCE_LENGTH = 64
DROPOUT_PROBABILITY = 0.1
POSITIONAL_EMBEDDING_OPTION = TextSinusoidalPositionalEmbeddingConfig

# Attention Options
ATTN_NUM_HEADS: int = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False
ATTN_ZERO_ATTENTION_FLAG: bool = False

## Attention Projection Stack Options
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = True
ATTN_STACK_HIDDEN_DIM: int = MODEL_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

### Attention Projection Gate Options
ATTN_STACK_GATE_FLAG: bool = False
ATTN_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#### Attention Projection Gate Stack Options
ATTN_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
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
ATTN_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
ATTN_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
ATTN_MEMORY_STACK_NUM_LAYERS: int | None = None
ATTN_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_MEMORY_STACK_BIAS_FLAG: bool | None = None

### Attention Projection Recurrent Layer Options
ATTN_RECURRENT_FLAG: bool = False
ATTN_RECURRENT_MAX_STEPS: int = 2
ATTN_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

#### Attention Projection Recurrent Gate Options
ATTN_RECURRENT_STACK_GATE_FLAG: bool = False
ATTN_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
ATTN_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

##### Attention Projection Recurrent Gate Stack Options
ATTN_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ATTN_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
ATTN_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ATTN_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None


# Feed-Forward Stack Options
FF_NUM_LAYERS: int = 1
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = 512
FF_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
FF_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_STACK_DROPOUT_PROBABILITY: float = DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

## Feed-Forward Gate Options
FF_STACK_GATE_FLAG: bool = False
FF_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

### Feed-Forward Gate Stack Options
FF_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_GATE_STACK_HIDDEN_DIM: int | None = None
FF_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_GATE_STACK_NUM_LAYERS: int | None = None
FF_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
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
FF_HALTING_STACK_NUM_LAYERS: int | None = None
FF_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
FF_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
FF_MEMORY_STACK_NUM_LAYERS: int | None = None
FF_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_MEMORY_STACK_BIAS_FLAG: bool | None = None

## Feed-Forward Recurrent Layer Options
FF_RECURRENT_FLAG: bool = False
FF_RECURRENT_MAX_STEPS: int = 2
FF_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

### Feed-Forward Recurrent Gate Options
FF_RECURRENT_STACK_GATE_FLAG: bool = False
FF_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
FF_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#### Feed-Forward Recurrent Gate Stack Options
FF_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
FF_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
FF_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

# Transformer Options
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
ENCODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
DECODER_LAYER_NORM_POSITION = LayerNormPositionOptions.BEFORE
ENCODER_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_SELF_ATTN_NUM_HEADS = ATTN_NUM_HEADS
DECODER_CROSS_ATTN_NUM_HEADS = ATTN_NUM_HEADS
ENCODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
DECODER_FEED_FORWARD_HIDDEN_DIM = FF_STACK_HIDDEN_DIM
ENCODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS
DECODER_FEED_FORWARD_NUM_LAYERS = FF_NUM_LAYERS

# Controller Options
STACK_GATE_FLAG = False
STACK_HALTING_FLAG = False
MEMORY_FLAG = False
RECURRENT_FLAG = False
RECURRENT_STACK_GATE_FLAG = False
RECURRENT_STACK_HALTING_FLAG = False
RECURRENT_MAX_STEPS = 2
STACK_RESIDUAL_CONNECTION_OPTION = None
RECURRENT_RESIDUAL_CONNECTION_OPTION = None

# Adaptive Parameter Options
PROJECTION_ADAPTIVE_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
PROJECTION_ADAPTIVE_BIAS_OPTION: type[DynamicBiasConfig] | None = None
PROJECTION_ADAPTIVE_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
PROJECTION_ADAPTIVE_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None

FEED_FORWARD_ADAPTIVE_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
FEED_FORWARD_ADAPTIVE_BIAS_OPTION: type[DynamicBiasConfig] | None = None
FEED_FORWARD_ADAPTIVE_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
FEED_FORWARD_ADAPTIVE_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None

# Trainer
NUM_EPOCHS = 30
TRAINER_DETERMINISTIC = True
TRAINER_GRADIENT_CLIP_VAL = 1.0
TRAINER_GRADIENT_CLIP_ALGORITHM = "norm"
TRAINER_LOG_EVERY_N_STEPS = 10

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
SEED = 0
