from emperor.embedding.absolute import (
    AbsolutePositionalEmbeddingConfig,
    TextLearnedPositionalEmbeddingConfig,
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
INPUT_DIM: int = 28782
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 28782
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
CONFIG_OVERRIDE_SKIP_KEYS: set[str] = {"HALTING_OUTPUT_DIM"}
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
CALLBACK_EARLY_STOPPING_PATIENCE: int = 5
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/loss"
CALLBACK_EARLY_STOPPING_MIN_DELTA: float = 0.0
CALLBACK_EARLY_STOPPING_STRICT: bool = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: bool = True
CALLBACK_CHECKPOINT_FLAG: bool = False

#########################################################################
# POSITIONAL EMBEDDING (added to token embeddings before the decoder)
POSITIONAL_EMBEDDING_OPTION: type[AbsolutePositionalEmbeddingConfig] = (
    TextLearnedPositionalEmbeddingConfig
)
POSITIONAL_EMBEDDING_PADDING_IDX: int = 0
POSITIONAL_EMBEDDING_AUTO_EXPAND_FLAG: bool = False

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_DROPOUT_PROBABILITY: float = 0.0
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True
#########################################################################
# EMBEDDING PIPELINE (applied to the summed token+positional embedding)
EMBEDDING_LAYER_NORM_FLAG: bool = False
EMBEDDING_DROPOUT_PROBABILITY: float = 0.1

#########################################################################
# LANGUAGE-MODELING HEAD
LM_HEAD_BIAS_FLAG: bool = False
LM_HEAD_WEIGHT_TYING_FLAG: bool = True

#########################################################################
# Layer Stack Submodule Options
SUBMODULE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
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
STACK_GATE_FLAG: bool = False
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
RECURRENT_STACK_GATE_FLAG: bool = False
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
ATTN_BIAS_FLAG: bool = True
ATTN_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ATTN_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
ATTN_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ATTN_STACK_DROPOUT_PROBABILITY: float = 0.0
ATTN_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
ATTN_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
### Attention Projection Gate Options
ATTN_STACK_GATE_FLAG: bool = False
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
ATTN_RECURRENT_STACK_GATE_FLAG: bool = False
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
FF_NUM_LAYERS: int = 1
FF_BIAS_FLAG: bool = True
FF_STACK_HIDDEN_DIM: int = HIDDEN_DIM * 4
FF_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
FF_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
FF_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
FF_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
FF_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
## Feed-Forward Gate Options
FF_STACK_GATE_FLAG: bool = False
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
FF_RECURRENT_STACK_GATE_FLAG: bool = False
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
FF_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
FF_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
FF_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
FF_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
FF_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
FF_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
FF_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None
