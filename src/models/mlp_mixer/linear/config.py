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
    DynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    MemoryPositionOptions,
)

#########################################################################
# Global
BATCH_SIZE: int = 64
INPUT_DIM: int = 224 * 224 * 3
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 10
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10

#########################################################################
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

# Callback
CALLBACK_EARLY_STOPPING_PATIENCE: int = 0
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"
CALLBACK_EARLY_STOPPING_MIN_DELTA: float = 0.0
CALLBACK_EARLY_STOPPING_STRICT: bool = True
CALLBACK_EARLY_STOPPING_CHECK_FINITE: bool = True
CALLBACK_CHECKPOINT_FLAG: bool = False

#########################################################################
# Layer Stack Options
# - hidden_dim comes from the global HIDDEN_DIM field above.
# - Token and channel mixer stacks inherit applicable activation/bias defaults.
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 8
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
STACK_DROPOUT_PROBABILITY: float = 0.0
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True

## Mixer Block Options
MIXER_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.RESIDUAL
)

#########################################################################
# Layer Stack Submodule Options
SUBMODULE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
SUBMODULE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
SUBMODULE_STACK_NUM_LAYERS: int = 1
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
# - If `STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
STACK_GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Gate Stack Options
# - If False, gate model stack options inherit the layer stack submodule options.
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
# - If `STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
STACK_HALTING_FLAG: bool = False
HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
HALTING_THRESHOLD: float = 0.999
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
## Halting Stack Options
# - If False, halting model stack options inherit the layer stack submodule options.
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
# - If `MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
MEMORY_FLAG: bool = False
MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Memory Stack Options
# - If False, memory model stack options inherit the layer stack submodule options.
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
# - If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 2
RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
RECURRENT_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
## Recurrent Gate Options
RECURRENT_STACK_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
### Recurrent Gate Stack Options
# - If False, recurrent gate stack options inherit gate/submodule stack options.
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
RECURRENT_HALTING_THRESHOLD: float = 0.999
RECURRENT_HALTING_DROPOUT: float = 0.0
RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
### Recurrent Halting Stack Options
# - If False, recurrent halting stack options inherit halting/submodule stack options.
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
## Recurrent Memory Options
RECURRENT_MEMORY_FLAG: bool = False

#########################################################################
# MLP Mixer Options
## Controller Stack Compatibility Options
# - These aliases mirror the canonical layer stack submodule options.
CONTROLLER_STACK_HIDDEN_DIM: int = SUBMODULE_STACK_HIDDEN_DIM
CONTROLLER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    SUBMODULE_STACK_LAYER_NORM_POSITION
)
CONTROLLER_STACK_NUM_LAYERS: int = SUBMODULE_STACK_NUM_LAYERS
CONTROLLER_STACK_ACTIVATION: ActivationOptions = SUBMODULE_STACK_ACTIVATION
CONTROLLER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
)
CONTROLLER_STACK_DROPOUT_PROBABILITY: float = SUBMODULE_STACK_DROPOUT_PROBABILITY
CONTROLLER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
CONTROLLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = (
    SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
CONTROLLER_STACK_BIAS_FLAG: bool = SUBMODULE_STACK_BIAS_FLAG

## Output Layer Options
OUTPUT_BIAS_FLAG: bool = True

## Image Patch Options
IMAGE_PATCH_SIZE: int = 16
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 224
PATCH_DROPOUT_PROBABILITY: float = 0.0
PATCH_BIAS_FLAG: bool = True

## Token Mixer Stack Options
# - STACK_ACTIVATION and STACK_BIAS_FLAG supply inherited branch defaults.
TOKEN_MIXER_STACK_HIDDEN_DIM: int = 64
TOKEN_MIXER_NUM_LAYERS: int = 2
TOKEN_MIXER_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
TOKEN_MIXER_STACK_DROPOUT_PROBABILITY: float = 0.0
TOKEN_MIXER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
TOKEN_MIXER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
TOKEN_MIXER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
TOKEN_MIXER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
TOKEN_MIXER_BIAS_FLAG: bool = STACK_BIAS_FLAG

## Channel Mixer Stack Options
# - STACK_ACTIVATION and STACK_BIAS_FLAG supply inherited branch defaults.
CHANNEL_MIXER_STACK_HIDDEN_DIM: int = 128
CHANNEL_MIXER_NUM_LAYERS: int = 2
CHANNEL_MIXER_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
CHANNEL_MIXER_STACK_DROPOUT_PROBABILITY: float = 0.0
CHANNEL_MIXER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
CHANNEL_MIXER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
CHANNEL_MIXER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
CHANNEL_MIXER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
CHANNEL_MIXER_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# Token Mixer Gate Options
# - If `TOKEN_MIXER_STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
TOKEN_MIXER_STACK_GATE_FLAG: bool = False
TOKEN_MIXER_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
TOKEN_MIXER_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Token Mixer Gate Stack Options
# - If False, gate stack options inherit the token mixer stack options.
TOKEN_MIXER_GATE_STACK_INDEPENDENT_FLAG: bool = False
TOKEN_MIXER_GATE_STACK_HIDDEN_DIM: int | None = None
TOKEN_MIXER_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
TOKEN_MIXER_GATE_STACK_NUM_LAYERS: int | None = None
TOKEN_MIXER_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
TOKEN_MIXER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
TOKEN_MIXER_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
TOKEN_MIXER_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
TOKEN_MIXER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
TOKEN_MIXER_GATE_STACK_BIAS_FLAG: bool | None = True

#########################################################################
# Token Mixer Halting Options
# - If `TOKEN_MIXER_STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
TOKEN_MIXER_STACK_HALTING_FLAG: bool = False
TOKEN_MIXER_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
TOKEN_MIXER_HALTING_THRESHOLD: float = 0.999
TOKEN_MIXER_HALTING_DROPOUT: float = 0.0
TOKEN_MIXER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
## Token Mixer Halting Stack Options
# - If False, halting stack options inherit the token mixer stack options.
TOKEN_MIXER_HALTING_STACK_INDEPENDENT_FLAG: bool = False
TOKEN_MIXER_HALTING_STACK_HIDDEN_DIM: int | None = None
TOKEN_MIXER_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
TOKEN_MIXER_HALTING_STACK_NUM_LAYERS: int | None = None
TOKEN_MIXER_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
TOKEN_MIXER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
TOKEN_MIXER_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
TOKEN_MIXER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
TOKEN_MIXER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
TOKEN_MIXER_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Token Mixer Memory Options
# - If `TOKEN_MIXER_MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
TOKEN_MIXER_MEMORY_FLAG: bool = False
TOKEN_MIXER_MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
TOKEN_MIXER_MEMORY_POSITION_OPTION: MemoryPositionOptions = (
    MemoryPositionOptions.AFTER_AFFINE
)
TOKEN_MIXER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
TOKEN_MIXER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Token Mixer Memory Stack Options
# - If False, memory stack options inherit the token mixer stack options.
TOKEN_MIXER_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
TOKEN_MIXER_MEMORY_STACK_HIDDEN_DIM: int | None = None
TOKEN_MIXER_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
TOKEN_MIXER_MEMORY_STACK_NUM_LAYERS: int | None = None
TOKEN_MIXER_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
TOKEN_MIXER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
TOKEN_MIXER_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
TOKEN_MIXER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
TOKEN_MIXER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
TOKEN_MIXER_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Token Mixer Recurrent Layer Options
# - If `TOKEN_MIXER_RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
TOKEN_MIXER_RECURRENT_FLAG: bool = False
TOKEN_MIXER_RECURRENT_MAX_STEPS: int = 2
TOKEN_MIXER_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
TOKEN_MIXER_RECURRENT_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
## Token Mixer Recurrent Gate Options
TOKEN_MIXER_RECURRENT_STACK_GATE_FLAG: bool = False
TOKEN_MIXER_RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
TOKEN_MIXER_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = (
    ActivationOptions.SIGMOID
)
### Token Mixer Recurrent Gate Stack Options
# - If False, recurrent gate stack options inherit the token mixer stack options.
TOKEN_MIXER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
TOKEN_MIXER_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
TOKEN_MIXER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
TOKEN_MIXER_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
TOKEN_MIXER_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
TOKEN_MIXER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
TOKEN_MIXER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
TOKEN_MIXER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
TOKEN_MIXER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
TOKEN_MIXER_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None
## Token Mixer Recurrent Halting Options
TOKEN_MIXER_RECURRENT_STACK_HALTING_FLAG: bool = False
TOKEN_MIXER_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
TOKEN_MIXER_RECURRENT_HALTING_THRESHOLD: float = 0.999
TOKEN_MIXER_RECURRENT_HALTING_DROPOUT: float = 0.0
TOKEN_MIXER_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
### Token Mixer Recurrent Halting Stack Options
# - If False, recurrent halting stack options inherit the token mixer stack options.
TOKEN_MIXER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
TOKEN_MIXER_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
TOKEN_MIXER_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Channel Mixer Gate Options
# - If `CHANNEL_MIXER_STACK_GATE_FLAG` is False, the gate-specific parameters below are ignored.
CHANNEL_MIXER_STACK_GATE_FLAG: bool = False
CHANNEL_MIXER_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
CHANNEL_MIXER_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
## Channel Mixer Gate Stack Options
# - If False, gate stack options inherit the channel mixer stack options.
CHANNEL_MIXER_GATE_STACK_INDEPENDENT_FLAG: bool = False
CHANNEL_MIXER_GATE_STACK_HIDDEN_DIM: int | None = None
CHANNEL_MIXER_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
CHANNEL_MIXER_GATE_STACK_NUM_LAYERS: int | None = None
CHANNEL_MIXER_GATE_STACK_ACTIVATION: ActivationOptions | None = ActivationOptions.TANH
CHANNEL_MIXER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
CHANNEL_MIXER_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
CHANNEL_MIXER_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
CHANNEL_MIXER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
CHANNEL_MIXER_GATE_STACK_BIAS_FLAG: bool | None = True

#########################################################################
# Channel Mixer Halting Options
# - If `CHANNEL_MIXER_STACK_HALTING_FLAG` is False, the halting-specific parameters below are ignored.
CHANNEL_MIXER_STACK_HALTING_FLAG: bool = False
CHANNEL_MIXER_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
CHANNEL_MIXER_HALTING_THRESHOLD: float = 0.999
CHANNEL_MIXER_HALTING_DROPOUT: float = 0.0
CHANNEL_MIXER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
## Channel Mixer Halting Stack Options
# - If False, halting stack options inherit the channel mixer stack options.
CHANNEL_MIXER_HALTING_STACK_INDEPENDENT_FLAG: bool = False
CHANNEL_MIXER_HALTING_STACK_HIDDEN_DIM: int | None = None
CHANNEL_MIXER_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
CHANNEL_MIXER_HALTING_STACK_NUM_LAYERS: int | None = None
CHANNEL_MIXER_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
CHANNEL_MIXER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
CHANNEL_MIXER_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
CHANNEL_MIXER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    LastLayerBiasOptions.DISABLED
)
CHANNEL_MIXER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
CHANNEL_MIXER_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Channel Mixer Memory Options
# - If `CHANNEL_MIXER_MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
CHANNEL_MIXER_MEMORY_FLAG: bool = False
CHANNEL_MIXER_MEMORY_OPTION: type[DynamicMemoryConfig] = (
    GatedResidualDynamicMemoryConfig
)
CHANNEL_MIXER_MEMORY_POSITION_OPTION: MemoryPositionOptions = (
    MemoryPositionOptions.AFTER_AFFINE
)
CHANNEL_MIXER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
CHANNEL_MIXER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
## Channel Mixer Memory Stack Options
# - If False, memory stack options inherit the channel mixer stack options.
CHANNEL_MIXER_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
CHANNEL_MIXER_MEMORY_STACK_HIDDEN_DIM: int | None = None
CHANNEL_MIXER_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
CHANNEL_MIXER_MEMORY_STACK_NUM_LAYERS: int | None = None
CHANNEL_MIXER_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
CHANNEL_MIXER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
CHANNEL_MIXER_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
CHANNEL_MIXER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
CHANNEL_MIXER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
CHANNEL_MIXER_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# Channel Mixer Recurrent Layer Options
# - If `CHANNEL_MIXER_RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
CHANNEL_MIXER_RECURRENT_FLAG: bool = False
CHANNEL_MIXER_RECURRENT_MAX_STEPS: int = 2
CHANNEL_MIXER_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
CHANNEL_MIXER_RECURRENT_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = (
    None
)
## Channel Mixer Recurrent Gate Options
CHANNEL_MIXER_RECURRENT_STACK_GATE_FLAG: bool = False
CHANNEL_MIXER_RECURRENT_GATE_OPTION: LayerGateOptions | None = (
    LayerGateOptions.MULTIPLIER
)
CHANNEL_MIXER_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = (
    ActivationOptions.SIGMOID
)
### Channel Mixer Recurrent Gate Stack Options
# - If False, recurrent gate stack options inherit the channel mixer stack options.
CHANNEL_MIXER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
CHANNEL_MIXER_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
CHANNEL_MIXER_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None
## Channel Mixer Recurrent Halting Options
CHANNEL_MIXER_RECURRENT_STACK_HALTING_FLAG: bool = False
CHANNEL_MIXER_RECURRENT_HALTING_OPTION: type[HaltingConfig] = StickBreakingConfig
CHANNEL_MIXER_RECURRENT_HALTING_THRESHOLD: float = 0.999
CHANNEL_MIXER_RECURRENT_HALTING_DROPOUT: float = 0.0
CHANNEL_MIXER_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
### Channel Mixer Recurrent Halting Stack Options
# - If False, recurrent halting stack options inherit the channel mixer stack options.
CHANNEL_MIXER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
CHANNEL_MIXER_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: (
    LayerNormPositionOptions | None
) = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: (
    LastLayerBiasOptions | None
) = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
CHANNEL_MIXER_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None
