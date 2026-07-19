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
from emperor.datasets.text.bert_pretraining import BERT_PRETRAINING_TARGET_VOCAB_SIZE
from emperor.embedding.absolute.core.config import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
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

# Global
INPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
    "HALTING_OUTPUT_DIM",
}
SEQUENCE_LENGTH: int = 35

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
DATA_NUM_WORKERS: int = 4
RUN_TEST_AFTER_FIT: bool = True
CALLBACK_EARLY_STOPPING_PATIENCE: int = 0
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"
CALLBACK_EARLY_STOPPING_MIN_DELTA: float = 0.0
CALLBACK_EARLY_STOPPING_STRICT: bool = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: bool = True
CALLBACK_CHECKPOINT_FLAG: bool = False

#########################################################################
# POSITIONAL EMBEDDING (added to token embeddings before the encoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    TextLearnedPositionalEmbeddingConfig
)
POSITIONAL_EMBEDDING_PADDING_IDX: int = 0
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool = False

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: int = 2
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.0
STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT
LAYER_NORM_POSITION: LayerNormPositionOptions = STACK_LAYER_NORM_POSITION
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True
CAUSAL_ATTENTION_MASK_FLAG: bool = False

#########################################################################
# EMBEDDING PIPELINE (applied to the summed token+positional+segment embedding)
TOKEN_TYPE_VOCAB_SIZE: int = 2
EMBEDDING_LAYER_NORM_FLAG: bool = True
EMBEDDING_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY

#########################################################################
# MASKED-LANGUAGE-MODELLING HEAD
MLM_ACTIVATION: ActivationOptions = ActivationOptions.GELU
MLM_DENSE_BIAS_FLAG: bool = True
MLM_LAYER_NORM_FLAG: bool = True
MLM_DECODER_BIAS_FLAG: bool = True
MLM_DECODER_WEIGHT_TYING_FLAG: bool = True

#########################################################################
# NEXT-SENTENCE-PREDICTION HEAD
NSP_POOLER_ACTIVATION: ActivationOptions = ActivationOptions.TANH
NSP_POOLER_BIAS_FLAG: bool = True
NSP_OUTPUT_DIM: int = 2
NSP_HEAD_BIAS_FLAG: bool = True

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
    LayerNormPositionOptions.BEFORE
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
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: bool = True

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
# Gate Options
GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Gate Stack Options
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
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
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
MEMORY_FLAG: bool = False
MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Memory Stack Options
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
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

## Recurrent Gate Options
RECURRENT_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
### Recurrent Gate Stack Options
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

## Recurrent Halting Options
RECURRENT_HALTING_FLAG: bool = False
RECURRENT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
RECURRENT_HALTING_DROPOUT: float = HALTING_DROPOUT
RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
### Recurrent Halting Stack Options
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
# Attention Options
ATTN_NUM_HEADS: int = 4
ATTN_ADD_KEY_VALUE_BIAS_FLAG: bool = False

## Attention Projection Stack Options
ATTN_NUM_LAYERS: int = 1
ATTN_BIAS_FLAG: bool = False
ATTN_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

#########################################################################
# Attention Projection Adaptive Parameter Options
### Attention Projection Adaptive Weight Options
ATTN_WEIGHT_OPTION_FLAG: bool = WEIGHT_OPTION_FLAG
ATTN_WEIGHT_OPTION: type[DynamicWeightConfig] | None = WEIGHT_OPTION
ATTN_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
ATTN_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
ATTN_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
ATTN_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
ATTN_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
ATTN_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WEIGHT_NORMALIZATION_POSITION_OPTION
)
ATTN_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
#### Attention Projection Weight Generator Stack Options
ATTN_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = (
    WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
)
ATTN_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: int | None = WEIGHT_GENERATOR_STACK_HIDDEN_DIM
ATTN_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
)
ATTN_WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = WEIGHT_GENERATOR_STACK_NUM_LAYERS
ATTN_WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    WEIGHT_GENERATOR_STACK_ACTIVATION
)
ATTN_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
)
ATTN_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ATTN_WEIGHT_GENERATOR_STACK_BIAS_FLAG: bool | None = WEIGHT_GENERATOR_STACK_BIAS_FLAG

### Attention Projection Adaptive Bias Options
ATTN_BIAS_OPTION_FLAG: bool = BIAS_OPTION_FLAG
ATTN_BIAS_OPTION: type[DynamicBiasConfig] | None = BIAS_OPTION
ATTN_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
ATTN_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
ATTN_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
ATTN_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = BIAS_BANK_EXPANSION_FACTOR
#### Attention Projection Bias Generator Stack Options
ATTN_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: bool = BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_BIAS_GENERATOR_STACK_HIDDEN_DIM: int | None = BIAS_GENERATOR_STACK_HIDDEN_DIM
ATTN_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
)
ATTN_BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = BIAS_GENERATOR_STACK_NUM_LAYERS
ATTN_BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    BIAS_GENERATOR_STACK_ACTIVATION
)
ATTN_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
)
ATTN_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ATTN_BIAS_GENERATOR_STACK_BIAS_FLAG: bool | None = BIAS_GENERATOR_STACK_BIAS_FLAG

### Attention Projection Adaptive Diagonal Options
ATTN_DIAGONAL_OPTION_FLAG: bool = DIAGONAL_OPTION_FLAG
ATTN_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = DIAGONAL_OPTION
#### Attention Projection Diagonal Generator Stack Options
ATTN_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: bool = (
    DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
)
ATTN_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: int | None = (
    DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
)
ATTN_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
)
ATTN_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = (
    DIAGONAL_GENERATOR_STACK_NUM_LAYERS
)
ATTN_DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    DIAGONAL_GENERATOR_STACK_ACTIVATION
)
ATTN_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
)
ATTN_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ATTN_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: bool | None = (
    DIAGONAL_GENERATOR_STACK_BIAS_FLAG
)

### Attention Projection Adaptive Mask Options
ATTN_MASK_OPTION_FLAG: bool = MASK_OPTION_FLAG
ATTN_ROW_MASK_OPTION: type[AxisMaskConfig] | None = ROW_MASK_OPTION
ATTN_MASK_THRESHOLD: float = MASK_THRESHOLD
ATTN_MASK_FLOOR: float = MASK_FLOOR
ATTN_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
ATTN_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
ATTN_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION
#### Attention Projection Mask Generator Stack Options
ATTN_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: bool = MASK_GENERATOR_STACK_INDEPENDENT_FLAG
ATTN_MASK_GENERATOR_STACK_HIDDEN_DIM: int | None = MASK_GENERATOR_STACK_HIDDEN_DIM
ATTN_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    MASK_GENERATOR_STACK_LAYER_NORM_POSITION
)
ATTN_MASK_GENERATOR_STACK_NUM_LAYERS: int | None = MASK_GENERATOR_STACK_NUM_LAYERS
ATTN_MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    MASK_GENERATOR_STACK_ACTIVATION
)
ATTN_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
ATTN_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
)
ATTN_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ATTN_MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = MASK_GENERATOR_STACK_BIAS_FLAG

#########################################################################
### Attention Projection Gate Options
ATTN_GATE_FLAG: bool = False
ATTN_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
ATTN_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
#### Attention Projection Gate Stack Options
ATTN_GATE_STACK_INDEPENDENT_FLAG: bool = False
ATTN_GATE_STACK_HIDDEN_DIM: int | None = None
ATTN_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ATTN_GATE_STACK_NUM_LAYERS: int | None = None
ATTN_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
ATTN_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ATTN_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
### Attention Projection Halting Options
ATTN_HALTING_FLAG: bool = False
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
ATTN_HALTING_STACK_NUM_LAYERS: int | None = None
ATTN_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
ATTN_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
ATTN_MEMORY_STACK_NUM_LAYERS: int | None = None
ATTN_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ATTN_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ATTN_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ATTN_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ATTN_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
### Attention Projection Recurrent Layer Options
ATTN_RECURRENT_FLAG: bool = False
ATTN_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
ATTN_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
#### Attention Projection Recurrent Gate Options
ATTN_RECURRENT_GATE_FLAG: bool = False
ATTN_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
ATTN_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
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

#########################################################################
#### Attention Projection Recurrent Halting Options
ATTN_RECURRENT_HALTING_FLAG: bool = False
ATTN_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
ATTN_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
ATTN_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
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

#########################################################################
# Feed-Forward Stack Options
FF_NUM_LAYERS: int = 2
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = HIDDEN_DIM
FF_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
FF_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
FF_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

#########################################################################
# Feed-Forward Adaptive Parameter Options
## Feed-Forward Adaptive Weight Options
FF_WEIGHT_OPTION_FLAG: bool = WEIGHT_OPTION_FLAG
FF_WEIGHT_OPTION: type[DynamicWeightConfig] | None = WEIGHT_OPTION
FF_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
FF_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
FF_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
FF_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
FF_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = WEIGHT_NORMALIZATION_OPTION
FF_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WEIGHT_NORMALIZATION_POSITION_OPTION
)
FF_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
### Feed-Forward Weight Generator Stack Options
FF_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = (
    WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG
)
FF_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: int | None = WEIGHT_GENERATOR_STACK_HIDDEN_DIM
FF_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION
)
FF_WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = WEIGHT_GENERATOR_STACK_NUM_LAYERS
FF_WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    WEIGHT_GENERATOR_STACK_ACTIVATION
)
FF_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY
)
FF_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
FF_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
FF_WEIGHT_GENERATOR_STACK_BIAS_FLAG: bool | None = WEIGHT_GENERATOR_STACK_BIAS_FLAG

## Feed-Forward Adaptive Bias Options
FF_BIAS_OPTION_FLAG: bool = BIAS_OPTION_FLAG
FF_BIAS_OPTION: type[DynamicBiasConfig] | None = BIAS_OPTION
FF_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
FF_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
FF_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
FF_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = BIAS_BANK_EXPANSION_FACTOR
### Feed-Forward Bias Generator Stack Options
FF_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: bool = BIAS_GENERATOR_STACK_INDEPENDENT_FLAG
FF_BIAS_GENERATOR_STACK_HIDDEN_DIM: int | None = BIAS_GENERATOR_STACK_HIDDEN_DIM
FF_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    BIAS_GENERATOR_STACK_LAYER_NORM_POSITION
)
FF_BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = BIAS_GENERATOR_STACK_NUM_LAYERS
FF_BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    BIAS_GENERATOR_STACK_ACTIVATION
)
FF_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
)
FF_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY
)
FF_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
FF_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
FF_BIAS_GENERATOR_STACK_BIAS_FLAG: bool | None = BIAS_GENERATOR_STACK_BIAS_FLAG

## Feed-Forward Adaptive Diagonal Options
FF_DIAGONAL_OPTION_FLAG: bool = DIAGONAL_OPTION_FLAG
FF_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = DIAGONAL_OPTION
### Feed-Forward Diagonal Generator Stack Options
FF_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: bool = (
    DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG
)
FF_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: int | None = DIAGONAL_GENERATOR_STACK_HIDDEN_DIM
FF_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION
)
FF_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = DIAGONAL_GENERATOR_STACK_NUM_LAYERS
FF_DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    DIAGONAL_GENERATOR_STACK_ACTIVATION
)
FF_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
FF_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY
)
FF_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
FF_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
FF_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: bool | None = DIAGONAL_GENERATOR_STACK_BIAS_FLAG

## Feed-Forward Adaptive Mask Options
FF_MASK_OPTION_FLAG: bool = MASK_OPTION_FLAG
FF_ROW_MASK_OPTION: type[AxisMaskConfig] | None = ROW_MASK_OPTION
FF_MASK_THRESHOLD: float = MASK_THRESHOLD
FF_MASK_FLOOR: float = MASK_FLOOR
FF_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
FF_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
FF_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION
### Feed-Forward Mask Generator Stack Options
FF_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: bool = MASK_GENERATOR_STACK_INDEPENDENT_FLAG
FF_MASK_GENERATOR_STACK_HIDDEN_DIM: int | None = MASK_GENERATOR_STACK_HIDDEN_DIM
FF_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    MASK_GENERATOR_STACK_LAYER_NORM_POSITION
)
FF_MASK_GENERATOR_STACK_NUM_LAYERS: int | None = MASK_GENERATOR_STACK_NUM_LAYERS
FF_MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = (
    MASK_GENERATOR_STACK_ACTIVATION
)
FF_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION
)
FF_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = (
    MASK_GENERATOR_STACK_DROPOUT_PROBABILITY
)
FF_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION
)
FF_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
FF_MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = MASK_GENERATOR_STACK_BIAS_FLAG

#########################################################################
## Feed-Forward Gate Options
FF_GATE_FLAG: bool = False
FF_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
FF_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
### Feed-Forward Gate Stack Options
FF_GATE_STACK_INDEPENDENT_FLAG: bool = False
FF_GATE_STACK_HIDDEN_DIM: int | None = None
FF_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
FF_GATE_STACK_NUM_LAYERS: int | None = None
FF_GATE_STACK_ACTIVATION: ActivationOptions | None = GATE_STACK_ACTIVATION
FF_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
FF_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
FF_GATE_STACK_BIAS_FLAG: bool | None = GATE_STACK_BIAS_FLAG

#########################################################################
## Feed-Forward Halting Options
FF_HALTING_FLAG: bool = False
FF_HALTING_THRESHOLD: float = HALTING_THRESHOLD
FF_HALTING_DROPOUT: float = HALTING_DROPOUT
FF_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = HALTING_HIDDEN_STATE_MODE
### Feed-Forward Halting Stack Options
FF_HALTING_STACK_INDEPENDENT_FLAG: bool = False
FF_HALTING_STACK_HIDDEN_DIM: int | None = None
FF_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    HALTING_STACK_LAYER_NORM_POSITION
)
FF_HALTING_STACK_NUM_LAYERS: int | None = None
FF_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
FF_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
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
FF_MEMORY_STACK_NUM_LAYERS: int | None = None
FF_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
FF_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
FF_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
## Feed-Forward Recurrent Layer Options
FF_RECURRENT_FLAG: bool = False
FF_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
FF_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
### Feed-Forward Recurrent Gate Options
FF_RECURRENT_GATE_FLAG: bool = False
FF_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
FF_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
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

#########################################################################
### Feed-Forward Recurrent Halting Options
FF_RECURRENT_HALTING_FLAG: bool = False
FF_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
FF_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
FF_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
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

# Workbench Config Schema Boundary
# Construction defaults remain available to flat CLI adapters and presets. The
# grouped builder interface exposes only top-level run settings in Workbench.
_PUBLIC_CONFIG_KEYS = {
    "INPUT_DIM",
    "OUTPUT_DIM",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
}
_PUBLIC_CONFIG_PREFIXES = (
    "TRAINER_",
    "CALLBACK_",
    "DATA_",
    "RUN_",
    "MONITOR_",
)
CONFIG_SCHEMA_SKIP_KEYS: set[str] = {
    key
    for key in globals()
    if key.isupper()
    and not key.startswith("_")
    and key != "CONFIG_SCHEMA_SKIP_KEYS"
    and key not in _PUBLIC_CONFIG_KEYS
    and not key.startswith(_PUBLIC_CONFIG_PREFIXES)
}
