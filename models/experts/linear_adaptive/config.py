from emperor.base.layer.residual import ResidualConnectionOptions
from models.trainer_config import *
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
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
from emperor.experiments.monitors import MonitorOption
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.base.layer.gate import LayerGateOptions
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,  # noqa: F401
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,  # noqa: F401
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,  # noqa: F401
)
from emperor.memory.options import MemoryPositionOptions
from emperor.augmentations.adaptive_parameters.options import (
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    AxisMaskConfig,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
)

# Global
BATCH_SIZE: int = 128
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 30
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_DEVICES: int = 1
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 10

# Callback
CALLBACK_EARLY_STOPPING_METRIC: str = "validation/accuracy"

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10

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
# ADAPTIVE GENERATOR STACK OPTIONS
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = STACK_HIDDEN_DIM
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    STACK_LAYER_NORM_POSITION
)
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.1
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_BIAS_FLAG: bool = STACK_BIAS_FLAG

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
# If False, router model stack options inherit the layer stack submodule options.
SAMPLER_STACK_INDEPENDENT_FLAG: bool = False
SAMPLER_STACK_HIDDEN_DIM: int | None = None
SAMPLER_STACK_NUM_LAYERS: int | None = None
SAMPLER_STACK_ACTIVATION: ActivationOptions | None = None
SAMPLER_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
SAMPLER_STACK_DROPOUT_PROBABILITY: float | None = None
SAMPLER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    LayerNormPositionOptions.DISABLED
)
SAMPLER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = True
SAMPLER_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER GATE OPTIONS
# If `ROUTER_GATE_FLAG` is False, the router gate parameters are ignored.
ROUTER_GATE_FLAG: bool = False
ROUTER_GATE_OPTION: LayerGateOptions | None = EXPERT_GATE_OPTION
ROUTER_GATE_ACTIVATION: ActivationOptions | None = EXPERT_GATE_ACTIVATION
# ROUTER GATE STACK OPTIONS
# If False, router gate stack options inherit router stack options.
ROUTER_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_GATE_STACK_ACTIVATION: ActivationOptions | None = EXPERT_GATE_STACK_ACTIVATION
ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = (
    EXPERT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)
ROUTER_GATE_STACK_BIAS_FLAG: bool | None = EXPERT_GATE_STACK_BIAS_FLAG

#########################################################################
# ROUTER HALTING OPTIONS
# If `ROUTER_HALTING_FLAG` is False, the router halting parameters are ignored.
ROUTER_HALTING_FLAG: bool = False
ROUTER_HALTING_THRESHOLD: float = EXPERT_HALTING_THRESHOLD
ROUTER_HALTING_DROPOUT: float = EXPERT_HALTING_DROPOUT
ROUTER_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    EXPERT_HALTING_HIDDEN_STATE_MODE
)
ROUTER_HALTING_OUTPUT_DIM: int = EXPERT_HALTING_OUTPUT_DIM
# ROUTER HALTING STACK OPTIONS
# If False, router halting stack options inherit router stack options.
ROUTER_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_HALTING_STACK_HIDDEN_DIM: int | None = None
ROUTER_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    EXPERT_HALTING_STACK_LAYER_NORM_POSITION
)
ROUTER_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION
)
ROUTER_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER MEMORY OPTIONS
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
# ROUTER MEMORY STACK OPTIONS
# If False, router memory stack options inherit router stack options.
ROUTER_MEMORY_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_MEMORY_STACK_HIDDEN_DIM: int | None = None
ROUTER_MEMORY_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_MEMORY_STACK_NUM_LAYERS: int | None = None
ROUTER_MEMORY_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions | None = None
ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MEMORY_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_MEMORY_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER RECURRENT LAYER OPTIONS
# If `ROUTER_RECURRENT_FLAG` is False, router recurrence is disabled.
ROUTER_RECURRENT_FLAG: bool = False
ROUTER_RECURRENT_MAX_STEPS: int = EXPERT_RECURRENT_MAX_STEPS
ROUTER_RECURRENT_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    EXPERT_RECURRENT_LAYER_NORM_POSITION
)

#########################################################################
# ROUTER RECURRENT GATE OPTIONS
ROUTER_RECURRENT_GATE_FLAG: bool = False
ROUTER_RECURRENT_GATE_OPTION: LayerGateOptions | None = EXPERT_RECURRENT_GATE_OPTION
ROUTER_RECURRENT_GATE_ACTIVATION: ActivationOptions | None = (
    EXPERT_RECURRENT_GATE_ACTIVATION
)
# ROUTER RECURRENT GATE STACK OPTIONS
# If False, router recurrent gate stack options inherit router stack options.
ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_GATE_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER RECURRENT HALTING OPTIONS
ROUTER_RECURRENT_HALTING_FLAG: bool = False
ROUTER_RECURRENT_HALTING_THRESHOLD: float = EXPERT_RECURRENT_HALTING_THRESHOLD
ROUTER_RECURRENT_HALTING_DROPOUT: float = EXPERT_RECURRENT_HALTING_DROPOUT
ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE
)
# ROUTER RECURRENT HALTING STACK OPTIONS
# If False, router recurrent halting stack options inherit router stack options.
ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS: int | None = None
ROUTER_RECURRENT_HALTING_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# WEIGHT GENERATOR OPTIONS
# If `WEIGHT_OPTION_FLAG` is False, the expert weight parameters below are ignored.
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
# WEIGHT GENERATOR STACK OPTIONS
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
# BIAS GENERATOR OPTIONS
# If `BIAS_OPTION_FLAG` is False, the expert bias parameters below are ignored.
BIAS_OPTION_FLAG: bool = False
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)
# BIAS GENERATOR STACK OPTIONS
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
# DIAGONAL GENERATOR OPTIONS
# If `DIAGONAL_OPTION_FLAG` is False, the expert diagonal parameters below are
# ignored.
DIAGONAL_OPTION_FLAG: bool = False
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# DIAGONAL GENERATOR STACK OPTIONS
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
# MASK OPTIONS
# If `MASK_OPTION_FLAG` is False, the expert mask parameters below are ignored.
MASK_OPTION_FLAG: bool = False
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1
MASK_SURROGATE_SCALE: float = 10.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN
# MASK STACK OPTIONS
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
# INPUT BOUNDARY PROJECTOR OPTIONS
# Input boundary dynamic weight options.
INPUT_LAYER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
INPUT_LAYER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
INPUT_LAYER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
INPUT_LAYER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
INPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
INPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
INPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = (
    WEIGHT_NORMALIZATION_POSITION_OPTION
)
INPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
# Input boundary dynamic bias options.
INPUT_LAYER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
INPUT_LAYER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
INPUT_LAYER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
INPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
INPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
# Input boundary dynamic diagonal options.
INPUT_LAYER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# Input boundary dynamic mask options.
INPUT_LAYER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
INPUT_LAYER_MASK_THRESHOLD: float = MASK_THRESHOLD
INPUT_LAYER_MASK_FLOOR: float = MASK_FLOOR
INPUT_LAYER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
INPUT_LAYER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
INPUT_LAYER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION

#########################################################################
# OUTPUT BOUNDARY PROJECTOR OPTIONS
# Output boundary dynamic weight options.
OUTPUT_LAYER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
OUTPUT_LAYER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
OUTPUT_LAYER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: WeightNormalizationPositionOptions = WEIGHT_NORMALIZATION_POSITION_OPTION
OUTPUT_LAYER_WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    WEIGHT_BANK_EXPANSION_FACTOR
)
# Output boundary dynamic bias options.
OUTPUT_LAYER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
OUTPUT_LAYER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
OUTPUT_LAYER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
OUTPUT_LAYER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
OUTPUT_LAYER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
# Output boundary dynamic diagonal options.
OUTPUT_LAYER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# Output boundary dynamic mask options.
OUTPUT_LAYER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
OUTPUT_LAYER_MASK_THRESHOLD: float = MASK_THRESHOLD
OUTPUT_LAYER_MASK_FLOOR: float = MASK_FLOOR
OUTPUT_LAYER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
OUTPUT_LAYER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
OUTPUT_LAYER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION

#########################################################################
# ROUTER WEIGHT GENERATOR OPTIONS
# If `ROUTER_WEIGHT_OPTION_FLAG` is False, router weight parameters are ignored.
ROUTER_WEIGHT_OPTION_FLAG: bool = False
ROUTER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
ROUTER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
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
# ROUTER WEIGHT GENERATOR STACK OPTIONS
# If False, router weight generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_WEIGHT_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_WEIGHT_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_WEIGHT_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_WEIGHT_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_WEIGHT_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_WEIGHT_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_WEIGHT_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_WEIGHT_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER BIAS GENERATOR OPTIONS
# If `ROUTER_BIAS_OPTION_FLAG` is False, router bias parameters are ignored.
ROUTER_BIAS_OPTION_FLAG: bool = False
ROUTER_BIAS_OPTION: type[DynamicBiasConfig] | None = None
ROUTER_BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = BIAS_DECAY_SCHEDULE
ROUTER_BIAS_DECAY_RATE: float = BIAS_DECAY_RATE
ROUTER_BIAS_DECAY_WARMUP_BATCHES: int = BIAS_DECAY_WARMUP_BATCHES
ROUTER_BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BIAS_BANK_EXPANSION_FACTOR
)
# ROUTER BIAS GENERATOR STACK OPTIONS
# If False, router bias generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_BIAS_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_BIAS_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_BIAS_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_BIAS_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_BIAS_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_BIAS_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_BIAS_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_BIAS_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER DIAGONAL GENERATOR OPTIONS
# If `ROUTER_DIAGONAL_OPTION_FLAG` is False, router diagonal parameters are ignored.
ROUTER_DIAGONAL_OPTION_FLAG: bool = False
ROUTER_DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# ROUTER DIAGONAL GENERATOR STACK OPTIONS
# If False, router diagonal generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_DIAGONAL_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_DIAGONAL_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = (
    None
)
ROUTER_DIAGONAL_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_DIAGONAL_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = (
    None
)
ROUTER_DIAGONAL_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_DIAGONAL_GENERATOR_STACK_BIAS_FLAG: bool | None = None

#########################################################################
# ROUTER MASK OPTIONS
# If `ROUTER_MASK_OPTION_FLAG` is False, router mask parameters are ignored.
ROUTER_MASK_OPTION_FLAG: bool = False
ROUTER_ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
ROUTER_MASK_THRESHOLD: float = MASK_THRESHOLD
ROUTER_MASK_FLOOR: float = MASK_FLOOR
ROUTER_MASK_TRANSITION_WIDTH: float = MASK_TRANSITION_WIDTH
ROUTER_MASK_SURROGATE_SCALE: float = MASK_SURROGATE_SCALE
ROUTER_MASK_DIMENSION_OPTION: MaskDimensionOptions = MASK_DIMENSION_OPTION
# ROUTER MASK STACK OPTIONS
# If False, router mask generator stack options inherit ADAPTIVE_GENERATOR_STACK_*.
ROUTER_MASK_GENERATOR_STACK_INDEPENDENT_FLAG: bool = False
ROUTER_MASK_GENERATOR_STACK_HIDDEN_DIM: int | None = None
ROUTER_MASK_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions | None = None
ROUTER_MASK_GENERATOR_STACK_NUM_LAYERS: int | None = None
ROUTER_MASK_GENERATOR_STACK_ACTIVATION: ActivationOptions | None = None
ROUTER_MASK_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions | None
) = None
ROUTER_MASK_GENERATOR_STACK_DROPOUT_PROBABILITY: float | None = None
ROUTER_MASK_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions | None = None
ROUTER_MASK_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool | None = None
ROUTER_MASK_GENERATOR_STACK_BIAS_FLAG: bool | None = None

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

#########################################################################
# DYNAMIC WEIGHT GENERATOR OPTION, DEPTH, DECAY, AND NORMALIZATION HYPERPARAMETERS
SEARCH_SPACE_WEIGHT_OPTION: list = [
    None,
    SingleModelDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
]
SEARCH_SPACE_WEIGHT_GENERATOR_DEPTH: list = [
    DynamicDepthOptions.DEPTH_OF_ONE,
    DynamicDepthOptions.DEPTH_OF_TWO,
    DynamicDepthOptions.DEPTH_OF_FOUR,
    DynamicDepthOptions.DEPTH_OF_SIX,
    DynamicDepthOptions.DEPTH_OF_EIGHT,
    DynamicDepthOptions.DEPTH_OF_TEN,
]
SEARCH_SPACE_WEIGHT_DECAY_SCHEDULE: list = [
    WeightDecayScheduleOptions.DISABLED,
    WeightDecayScheduleOptions.EXPONENTIAL,
    WeightDecayScheduleOptions.LINEAR,
    WeightDecayScheduleOptions.MULTIPLICATIVE,
]
SEARCH_SPACE_WEIGHT_DECAY_RATE: list = [1e-5, 1e-4, 1e-3, 1e-2]
SEARCH_SPACE_WEIGHT_DECAY_WARMUP_BATCHES: list = [0, 100, 500, 1000]
SEARCH_SPACE_WEIGHT_NORMALIZATION_OPTION: list = [
    WeightNormalizationOptions.DISABLED,
    WeightNormalizationOptions.CLAMP,
    WeightNormalizationOptions.L2_SCALE,
    WeightNormalizationOptions.SOFT_CLAMP,
    WeightNormalizationOptions.RMS,
    WeightNormalizationOptions.SIGMOID_SCALE,
]
SEARCH_SPACE_WEIGHT_NORMALIZATION_POSITION_OPTION: list = [
    WeightNormalizationPositionOptions.DISABLED,
    WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
    WeightNormalizationPositionOptions.AFTER_OUTER_PRODUCT,
]
SEARCH_SPACE_WEIGHT_BANK_EXPANSION_FACTOR: list = [
    BankExpansionFactorOptions.FACTOR_OF_ONE,
    BankExpansionFactorOptions.FACTOR_OF_TWO,
    BankExpansionFactorOptions.FACTOR_OF_THREE,
    BankExpansionFactorOptions.FACTOR_OF_FOUR,
]

#########################################################################
# DYNAMIC BIAS GENERATOR OPTION, DECAY, AND BANK EXPANSION HYPERPARAMETERS
SEARCH_SPACE_BIAS_OPTION: list = [
    None,
    AffineTransformDynamicBiasConfig,
    AdditiveDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
]
SEARCH_SPACE_BIAS_DECAY_SCHEDULE: list = [
    WeightDecayScheduleOptions.DISABLED,
    WeightDecayScheduleOptions.EXPONENTIAL,
    WeightDecayScheduleOptions.LINEAR,
    WeightDecayScheduleOptions.MULTIPLICATIVE,
]
SEARCH_SPACE_BIAS_DECAY_RATE: list = [1e-5, 1e-4, 1e-3, 1e-2]
SEARCH_SPACE_BIAS_DECAY_WARMUP_BATCHES: list = [0, 100, 500, 1000]
SEARCH_SPACE_BIAS_BANK_EXPANSION_FACTOR: list = [
    BankExpansionFactorOptions.FACTOR_OF_ONE,
    BankExpansionFactorOptions.FACTOR_OF_TWO,
    BankExpansionFactorOptions.FACTOR_OF_THREE,
    BankExpansionFactorOptions.FACTOR_OF_FOUR,
]

#########################################################################
# DYNAMIC DIAGONAL GENERATOR OPTION AND DEPTH HYPERPARAMETERS
SEARCH_SPACE_DIAGONAL_OPTION: list = [
    None,
    StandardDynamicDiagonalConfig,
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
]

#########################################################################
# DYNAMIC MASK GENERATOR OPTION, DEPTH, DIMENSION, AND SURROGATE SHAPING HYPERPARAMETERS
SEARCH_SPACE_ROW_MASK_OPTION: list = [
    None,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
]
SEARCH_SPACE_MASK_THRESHOLD: list = [0.1, 0.3, 0.5, 0.7, 0.9]
SEARCH_SPACE_MASK_SURROGATE_SCALE: list = [1.0, 5.0, 10.0, 20.0]
SEARCH_SPACE_MASK_FLOOR: list = [0.0, 0.1, 0.25, 0.5]
SEARCH_SPACE_MASK_TRANSITION_WIDTH: list = [0.05, 0.1, 0.2, 0.5]
SEARCH_SPACE_MASK_DIMENSION_OPTION: list = [
    MaskDimensionOptions.ROW,
    MaskDimensionOptions.COLUMN,
]

#########################################################################
# AUGMENTATION GENERATOR LAYER STACK HYPERPARAMETERS
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: list = [1, 2, 3]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.SILU,
    ActivationOptions.GELU,
    ActivationOptions.MISH,
]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
]

SEARCH_SPACE_ROUTER_WEIGHT_OPTION: list = SEARCH_SPACE_WEIGHT_OPTION
SEARCH_SPACE_ROUTER_BIAS_OPTION: list = SEARCH_SPACE_BIAS_OPTION
SEARCH_SPACE_ROUTER_DIAGONAL_OPTION: list = SEARCH_SPACE_DIAGONAL_OPTION
SEARCH_SPACE_ROUTER_ROW_MASK_OPTION: list = SEARCH_SPACE_ROW_MASK_OPTION

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
        name="adaptive",
        label="Adaptive parameters",
        description=(
            "Logs dynamic weight, bias, diagonal, and mask parameter statistics, "
            "plus input-adaptivity (cross-sample variation / collapse detection)."
        ),
        kinds=["scalar", "histogram"],
        callback_factory=lambda: AdaptiveParameterMonitorCallback(
            log_every_n_steps=100,
            log_histograms=True,
            log_internal_stats=True,
        ),
    ),
    MonitorOption(
        name="weight-bank",
        label="Weight bank utilization",
        description=(
            "Logs bank-slot selection entropy, utilization, and routing heatmaps "
            "for weighted-bank dynamic params."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: WeightBankUtilizationMonitorCallback(
            log_every_n_steps=100,
        ),
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
