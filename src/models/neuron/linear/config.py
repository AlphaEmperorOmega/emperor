# ruff: noqa: E402

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
from emperor.base.layer.gate import LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
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
BATCH_SIZE: int = 128
INPUT_DIM: int = 28**2
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 10
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30

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
# Neuron Wrapper Options
from emperor.neuron.core.optimizer_sync import NeuronClusterOptimizerSyncCallback
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions

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
CLUSTER_TERMINAL_ROUTER_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
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
CLUSTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
CLUSTER_HALTING_STACK_DROPOUT_PROBABILITY: float = 0.0
CLUSTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
CLUSTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
CLUSTER_HALTING_STACK_BIAS_FLAG: bool = True
