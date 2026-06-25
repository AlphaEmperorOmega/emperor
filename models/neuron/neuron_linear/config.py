from emperor.base.layer.residual import ResidualConnectionOptions

# Generated from models.linears.linear.config by scaffold_wrapper_model.py.
from models.trainer_config import *
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.experiments.monitors import MonitorOption
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.neuron.core.monitor import NeuronClusterMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback
from emperor.neuron.core.optimizer_sync import NeuronClusterOptimizerSyncCallback
from emperor.halting.core.monitor import HaltingMonitorCallback
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

# Global
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
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
            "per-neuron terminal samplers: probability distributions, "
            "per-expert utilization, and auxiliary load-balancing loss "
            "components."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: SamplerMonitorCallback(log_every_n_steps=100),
    ),
    MonitorOption(
        name="halting",
        label="Halting (adaptive compute)",
        description=(
            "Logs recurrence depth, halting fraction, max-steps saturation, ponder "
            "loss, plus survival heatmap and ponder-cost histogram for "
            "stick-breaking / soft halting modules."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: HaltingMonitorCallback(log_every_n_steps=100),
        default_enabled=True,
    ),
    MonitorOption(
        name="memory",
        label="Memory modules",
        description=(
            "Logs gating, blend-weight, and state statistics for Emperor memory "
            "modules. Inactive until a memory config is enabled (e.g. axons "
            "memory or layer memory)."
        ),
        kinds=["scalar"],
        callback_factory=lambda: MemoryMonitorCallback(log_every_n_steps=100),
    ),
]

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 10

# Callback
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"
CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC = NeuronClusterOptimizerSyncCallback()

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10

#########################################################################
# LAYER STACK OPTIONS
STACK_HIDDEN_DIM: int = 64
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 2
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = True

#########################################################################
# GATE STACK OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
GATE_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = 2
GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.TANH
GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    STACK_RESIDUAL_CONNECTION_OPTION
)
GATE_STACK_DROPOUT_PROBABILITY: float = 0.0
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = STACK_LAST_LAYER_BIAS_OPTION
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
GATE_STACK_BIAS_FLAG: bool = True

#########################################################################
# Halting options
# HALTING OPTIONS
# If `HALTING_FLAG` is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
HALTING_OUTPUT_DIM: int = 2
HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
HALTING_STACK_NUM_LAYERS: int = 2
HALTING_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
HALTING_STACK_DROPOUT_PROBABILITY: float = 0.0
HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

#########################################################################
# RECURRENT LAYER OPTIONS
# If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4

#########################################################################
# RECURRENT GATE STACK OPTIONS
RECURRENT_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID

#########################################################################
# RECURRENT HALTING OPTIONS
RECURRENT_HALTING_FLAG: bool = False

#########################################################################
# HYPERPARAMETER SEARCH SPACE
# These values define the parameter ranges explored when search mode is enabled.
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_STACK_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEARCH_SPACE_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.LEAKY_RELU,
    ActivationOptions.ELU,
    ActivationOptions.GELU,
    ActivationOptions.TANH,
]


#########################################################################
# NEURON WRAPPER OPTIONS
from emperor.base.options import ActivationOptions
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions

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

# Applies to the terminal sampler of every neuron; the cluster entry sampler is
# derived from this config with num_experts/top_k adjusted to the entry plane.
CLUSTER_TERMINAL_SAMPLER_THRESHOLD: float = 0.0
CLUSTER_TERMINAL_SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
CLUSTER_TERMINAL_SAMPLER_NUM_TOPK_SAMPLES: int = 0
CLUSTER_TERMINAL_SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_NOISY_TOPK_FLAG: bool = False
CLUSTER_TERMINAL_SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
CLUSTER_TERMINAL_SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

# CLUSTER HALTING OPTIONS
# If `CLUSTER_HALTING_FLAG` is False, the cluster-halting parameters below are ignored.
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
