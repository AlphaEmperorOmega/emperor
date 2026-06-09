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
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.sampler.core.monitor import SamplerMonitorCallback
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
        callback_factory=lambda: RecurrentLayerMonitorCallback(
            log_every_n_steps=100
        ),
    ),
    MonitorOption(
        name="layer-controller",
        label="Layer controllers",
        description=(
            "Logs Layer gate, residual, dropout, layer-norm, and activation "
            "controller statistics without duplicating memory metrics."
        ),
        kinds=["scalar"],
        callback_factory=lambda: LayerControllerMonitorCallback(
            log_every_n_steps=100
        ),
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
]

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
BIAS_FLAG: bool = True

#########################################################################
# MAIN STACK / HIDDEN DIM
HIDDEN_DIM: int = 256
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

#########################################################################
# GATE STACK OPTIONS (applied per outer MoE-layer in the main stack)
# If GATE_FLAG is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_HIDDEN_DIM: int = HIDDEN_DIM
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = 2
GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.TANH
GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
GATE_STACK_DROPOUT_PROBABILITY: float = 0.0
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = STACK_LAST_LAYER_BIAS_OPTION
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
GATE_BIAS_FLAG: bool = True

#########################################################################
# HALTING OPTIONS (applied per outer MoE-layer in the main stack)
# If HALTING_FLAG is False, the halting-specific parameters below are ignored.
HALTING_FLAG: bool = False
HALTING_THRESHOLD: float = 0.99
HALTING_DROPOUT: float = 0.0
HALTING_HIDDEN_STATE_MODE: HaltingHiddenStateModeOptions = (
    HaltingHiddenStateModeOptions.RAW
)
HALTING_HIDDEN_DIM: int = HIDDEN_DIM
HALTING_OUTPUT_DIM: int = 2
HALTING_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
HALTING_STACK_NUM_LAYERS: int = 2
HALTING_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
HALTING_STACK_RESIDUAL_FLAG: bool = False
HALTING_STACK_DROPOUT_PROBABILITY: float = 0.0
HALTING_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# RECURRENT LAYER OPTIONS
# If `RECURRENT_FLAG` is False, the recurrent-specific parameters below are ignored.
RECURRENT_FLAG: bool = False
RECURRENT_MAX_STEPS: int = 4
RECURRENT_GATE_FLAG: bool = False
RECURRENT_HALTING_FLAG: bool = False

#########################################################################
# MIXTURE-OF-EXPERTS SCALAR PARAMS
EXPERT_TOP_K: int = 2
EXPERT_NUM_EXPERTS: int = 4
EXPERT_CAPACITY_FACTOR: float = 0.0
EXPERT_DROPPED_TOKEN_BEHAVIOR: DroppedTokenOptions = DroppedTokenOptions.ZEROS
EXPERT_COMPUTE_EXPERT_MIXTURE_FLAG: bool = True
EXPERT_WEIGHTED_PARAMETERS_FLAG: bool = True
EXPERT_WEIGHTING_POSITION_OPTION: ExpertWeightingPositionOptions = (
    ExpertWeightingPositionOptions.BEFORE_EXPERTS
)
EXPERT_ROUTING_INITIALIZATION_MODE: RoutingInitializationMode = (
    RoutingInitializationMode.LAYER
)

#########################################################################
# EXPERT STACK
EXPERT_STACK_NUM_LAYERS: int = 2
EXPERT_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
EXPERT_STACK_RESIDUAL_FLAG: bool = False
EXPERT_STACK_DROPOUT_PROBABILITY: float = 0.0
EXPERT_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
EXPERT_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
EXPERT_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
EXPERT_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# SAMPLER SCALAR PARAMS
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = True
SAMPLER_NOISY_TOPK_FLAG: bool = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

#########################################################################
# ROUTER SCALAR PARAMS
ROUTER_NOISY_TOPK_FLAG: bool = False

#########################################################################
# SAMPLER ROUTER STACK
SAMPLER_STACK_NUM_LAYERS: int = 2
SAMPLER_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
SAMPLER_STACK_RESIDUAL_FLAG: bool = False
SAMPLER_STACK_DROPOUT_PROBABILITY: float = 0.0
SAMPLER_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
SAMPLER_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
SAMPLER_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
SAMPLER_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# ADAPTIVE WEIGHT OPTIONS (shared across input, output, expert, sampler slots)
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

#########################################################################
# ADAPTIVE BIAS OPTIONS
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)

#########################################################################
# ADAPTIVE DIAGONAL OPTIONS
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None

#########################################################################
# ADAPTIVE MASK OPTIONS
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1
MASK_SURROGATE_SCALE: float = 10.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN

#########################################################################
# AUGMENTATION GENERATOR STACK (shared)
ADAPTIVE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ADAPTIVE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
ADAPTIVE_STACK_NUM_LAYERS: int = 2
ADAPTIVE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_STACK_RESIDUAL_FLAG: bool = False
ADAPTIVE_STACK_DROPOUT_PROBABILITY: float = 0.1
ADAPTIVE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
# HYPERPARAMETER SEARCH SPACE
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
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
