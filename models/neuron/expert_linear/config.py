from emperor.base.layer.residual import ResidualConnectionOptions
from models.trainer_config import *
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.experiments.monitors import MonitorOption
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from emperor.memory.options import MemoryPositionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

# Global
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0


#########################################################################
# LAYER STACK OPTIONS
STACK_HIDDEN_DIM: int = 32
STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True

#########################################################################
# LAYER STACK SUBMODULE OPTIONS
SUBMODULE_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
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
# GATE OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
# GATE STACK OPTIONS
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
# HALTING OPTIONS
# If `HALTING_FLAG` is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_OUTPUT_DIM: int = 2
# HALTING STACK OPTIONS
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
# MEMORY OPTIONS
# If `MEMORY_FLAG` is False, the memory-specific parameters below are ignored.
MEMORY_FLAG: bool = False
MEMORY_OPTION: type[DynamicMemoryConfig] = GatedResidualDynamicMemoryConfig
MEMORY_POSITION_OPTION: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE
MEMORY_TEST_TIME_TRAINING_LEARNING_RATE: float | None = None
MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS: int | None = None
# MEMORY STACK OPTIONS
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
# RECURRENT LAYER OPTIONS
# If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)

#########################################################################
# RECURRENT GATE OPTIONS
RECURRENT_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
# RECURRENT GATE STACK OPTIONS
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
# RECURRENT HALTING OPTIONS
RECURRENT_HALTING_FLAG: bool = False
RECURRENT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
RECURRENT_HALTING_DROPOUT: float = HALTING_DROPOUT
RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
# RECURRENT HALTING STACK OPTIONS
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
# MIXTURE OF EXPERTS MODEL OPTIONS
EXPERT_TOP_K: int = 2
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
# EXPERT STACK OPTIONS
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
# EXPERT GATE OPTIONS
# If `EXPERT_GATE_FLAG` is False, the expert gate parameters below are ignored.
EXPERT_GATE_FLAG: bool = False
EXPERT_GATE_OPTION: LayerGateOptions | None = GATE_OPTION
EXPERT_GATE_ACTIVATION: ActivationOptions | None = GATE_ACTIVATION
# EXPERT GATE STACK OPTIONS
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
# EXPERT HALTING OPTIONS
# If `EXPERT_HALTING_FLAG` is False, the expert halting parameters are ignored.
EXPERT_HALTING_FLAG: bool = False
EXPERT_HALTING_THRESHOLD: float = HALTING_THRESHOLD
EXPERT_HALTING_DROPOUT: float = HALTING_DROPOUT
EXPERT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HALTING_HIDDEN_STATE_MODE
)
EXPERT_HALTING_OUTPUT_DIM: int = HALTING_OUTPUT_DIM
# EXPERT HALTING STACK OPTIONS
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
# EXPERT MEMORY OPTIONS
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
# EXPERT MEMORY STACK OPTIONS
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
# EXPERT RECURRENT LAYER OPTIONS
# If `EXPERT_RECURRENT_FLAG` is False, expert recurrence is disabled.
EXPERT_RECURRENT_FLAG: bool = False
EXPERT_RECURRENT_MAX_STEPS: int = RECURRENT_MAX_STEPS
EXPERT_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
# EXPERT RECURRENT GATE OPTIONS
EXPERT_RECURRENT_GATE_FLAG: bool = False
EXPERT_RECURRENT_GATE_OPTION: LayerGateOptions | None = RECURRENT_GATE_OPTION
EXPERT_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = RECURRENT_GATE_ACTIVATION
# EXPERT RECURRENT GATE STACK OPTIONS
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
# EXPERT RECURRENT HALTING OPTIONS
EXPERT_RECURRENT_HALTING_FLAG: bool = False
EXPERT_RECURRENT_HALTING_THRESHOLD: float = RECURRENT_HALTING_THRESHOLD
EXPERT_RECURRENT_HALTING_DROPOUT: float = RECURRENT_HALTING_DROPOUT
EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    RECURRENT_HALTING_HIDDEN_STATE_MODE
)
# EXPERT RECURRENT HALTING STACK OPTIONS
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
# SAMPLER MODEL OPTIONS
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
# ROUTER OPTIONS
ROUTER_NOISY_TOPK_FLAG: bool = False

#########################################################################
# ROUTER STACK OPTIONS
ROUTER_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
ROUTER_STACK_NUM_LAYERS: int = 2
ROUTER_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ROUTER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
ROUTER_STACK_DROPOUT_PROBABILITY: float = 0.0
ROUTER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    STACK_LAYER_NORM_POSITION
)
ROUTER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ROUTER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ROUTER_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# HYPERPARAMETER SEARCH SPACE
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_STACK_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEARCH_SPACE_STACK_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_LAYER_NORM_POSITION: list = SEARCH_SPACE_STACK_LAYER_NORM_POSITION
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.LEAKY_RELU,
    ActivationOptions.ELU,
    ActivationOptions.GELU,
    ActivationOptions.TANH,
]
MONITOR_OPTIONS: list[MonitorOption] = [
    MonitorOption(
        name="linear",
        label="Linear layers",
        description=(
            "Logs activation, parameter, gradient, weight-conditioning "
            "(spectral norm / condition number / effective rank), and dead-feature "
            "stats for Emperor linear layers."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LinearMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="recurrent-layer",
        label="Recurrent layers",
        description=(
            "Logs recurrent step count, hidden-state convergence, recurrent gate "
            "openness, halted-state preservation, and step-delta visual summaries."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: RecurrentLayerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="sampler",
        label="Sampler usage",
        description=(
            "Logs expert routing balance, capacity drop fraction, auxiliary "
            "load-balancing loss, usage histograms, and routing heatmaps."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description=(
            "Logs gating, blend-weight, and state statistics for Emperor memory "
            "modules. Inactive until a memory config is enabled."
        ),
        kinds=["scalar"],
        callback_factory=lambda: MemoryMonitorCallback(log_every_n_steps=100),
    ),
]

#########################################################################
# NEURON WRAPPER OPTIONS
from emperor.neuron.core.monitor import NeuronClusterMonitorCallback
from emperor.neuron.core.optimizer_sync import NeuronClusterOptimizerSyncCallback
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions
from emperor.sampler.core.monitor import SamplerMonitorCallback

CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC = NeuronClusterOptimizerSyncCallback()

_neuron_monitor_options = [
    MonitorOption(
        name="neuron_cluster",
        label="Neuron cluster growth",
        description=(
            "Logs cluster growth (count, capacity, fill, growth pressure) plus "
            "routing dynamics: route depth, escape/halt fractions, entry-routing "
            "entropy, survival curve, and per-neuron utilization heatmap."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: NeuronClusterMonitorCallback(log_every_n_steps=100),
        default_enabled=True,
    ),
    MonitorOption(
        name="sampler",
        label="Routing samplers",
        description=(
            "Logs router/sampler internals for the cluster entry sampler and "
            "per-neuron terminal samplers: probability distributions, per-expert "
            "utilization, and auxiliary load-balancing loss components."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
]
_existing_monitor_names = {option.name for option in MONITOR_OPTIONS}
MONITOR_OPTIONS = [
    *MONITOR_OPTIONS,
    *[
        option
        for option in _neuron_monitor_options
        if option.name not in _existing_monitor_names
    ],
]

CLUSTER_X_AXIS_TOTAL_NEURONS: int = 10
CLUSTER_Y_AXIS_TOTAL_NEURONS: int = 10
CLUSTER_Z_AXIS_TOTAL_NEURONS: int = 1
CLUSTER_INITIAL_X_AXIS_TOTAL_NEURONS: int = 3
CLUSTER_INITIAL_Y_AXIS_TOTAL_NEURONS: int = 3
CLUSTER_INITIAL_Z_AXIS_TOTAL_NEURONS: int = 1
CLUSTER_MAX_STEPS: int = 4
CLUSTER_GROWTH_THRESHOLD: int | None = 250

CLUSTER_TERMINAL_XY_AXIS_RANGE: TerminalRangeOptions = TerminalRangeOptions.ONE
CLUSTER_TERMINAL_Z_AXIS_RANGE: TerminalRangeOptions = TerminalRangeOptions.ONE
CLUSTER_TERMINAL_Z_AXIS_OFFSET: TerminalZAxisOffsetOptions = (
    TerminalZAxisOffsetOptions.ZERO
)
CLUSTER_TERMINAL_TOP_K: int = 1
CLUSTER_TERMINAL_ROUTER_NUM_LAYERS: int = 1
CLUSTER_TERMINAL_ROUTER_HIDDEN_DIM: int = STACK_HIDDEN_DIM
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

CLUSTER_TERMINAL_SAMPLER_THRESHOLD: float = 0.0
CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES: int = 0
CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

CLUSTER_HALTING_FLAG: bool = True
CLUSTER_HALTING_THRESHOLD: float = 0.95
CLUSTER_HALTING_DROPOUT: float = 0.0
CLUSTER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
CLUSTER_HALTING_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
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

SEARCH_SPACE_CLUSTER_MAX_STEPS: list = [1, 2, 4, 6]
SEARCH_SPACE_CLUSTER_TERMINAL_TOP_K: list = [1, 2]
SEARCH_SPACE_CLUSTER_GROWTH_THRESHOLD: list = [100, 250, 500, None]
