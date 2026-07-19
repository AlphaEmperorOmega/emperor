from emperor.datasets.text.bert_pretraining import BERT_PRETRAINING_TARGET_VOCAB_SIZE
from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
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
INPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = BERT_PRETRAINING_TARGET_VOCAB_SIZE
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {
    "BERT_PRETRAINING_TARGET_VOCAB_SIZE",
}
SEQUENCE_LENGTH: int = 35

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0

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
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
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
# MASKED-LANGUAGE-MODELING HEAD
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
SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
SUBMODULE_STACK_DROPOUT_PROBABILITY: float = 0.0
SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
SUBMODULE_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

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
HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
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
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

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
FF_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

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
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Mixture Of Experts Model Options
EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG: bool = False
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
EXPERT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
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
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_STACK_DROPOUT_PROBABILITY: float = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.BEFORE
)
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ROUTER_BIAS_FLAG: bool = True

# Workbench Config Schema Boundary
# Construction defaults remain available to flat CLI adapters and presets. The
# grouped builder interface exposes only top-level run settings in Workbench.
_PUBLIC_CONFIG_KEYS = {
    "INPUT_DIM",
    "OUTPUT_DIM",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "SEQUENCE_LENGTH",
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
