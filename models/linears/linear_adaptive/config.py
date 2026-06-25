from emperor.base.layer.residual import ResidualConnectionOptions
from models.trainer_config import *
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.datasets.image.classification.mnist import Mnist
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.experiments.monitors import MonitorOption
from emperor.base.layer.monitor import (
    LayerControllerMonitorCallback,
    RecurrentLayerMonitorCallback,
)
from emperor.base.layer.gate import LayerGateOptions
from emperor.halting.core.monitor import HaltingMonitorCallback
from emperor.augmentations.adaptive_parameters.core.monitor import (
    AdaptiveParameterMonitorCallback,
)
from emperor.augmentations.adaptive_parameters.core.bank_monitor import (
    WeightBankUtilizationMonitorCallback,
)
from emperor.linears.core.monitor import LinearMonitorCallback
from emperor.memory.core.monitor import MemoryMonitorCallback
from emperor.memory.config import (
    AttentionDynamicMemoryConfig,
    DynamicMemoryConfig,
    ElementWiseWeightedDynamicMemoryConfig,
    GatedResidualDynamicMemoryConfig,
    WeightedDynamicMemoryConfig,
)
from emperor.memory.options import MemoryPositionOptions
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
    WeightDecayScheduleOptions,
    MaskDimensionOptions,
    BankExpansionFactorOptions,
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
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
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
BIAS_FLAG: bool = True

#########################################################################
# LAYER STACK OPTIONS
HIDDEN_DIM: int = 256
STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 5
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
STACK_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# LAYER STACK SUBMODULE OPTIONS
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
SUBMODULE_STACK_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# ADAPTIVE SUBMODULE STACK OPTIONS
ADAPTIVE_SUBMODULE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ADAPTIVE_SUBMODULE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = STACK_LAYER_NORM_POSITION
ADAPTIVE_SUBMODULE_STACK_NUM_LAYERS: int = 2
ADAPTIVE_SUBMODULE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION: ResidualConnectionOptions = (
    ResidualConnectionOptions.DISABLED
)
ADAPTIVE_SUBMODULE_STACK_DROPOUT_PROBABILITY: float = 0.1
ADAPTIVE_SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
ADAPTIVE_SUBMODULE_STACK_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# GATE STACK OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
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
# If False, halting model stack options inherit the layer stack submodule options.
HALTING_STACK_INDEPENDENT_FLAG: bool = False
HALTING_STACK_HIDDEN_DIM: int | None = None
HALTING_OUTPUT_DIM: int = 2
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
# RECURRENT GATE STACK OPTIONS
RECURRENT_GATE_FLAG: bool = False
RECURRENT_GATE_OPTION: LayerGateOptions | None = LayerGateOptions.MULTIPLIER
RECURRENT_GATE_ACTIVATION: ActivationOptions | None = ActivationOptions.SIGMOID
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
# If False, recurrent halting stack options inherit halting/submodule stack options.
RECURRENT_HALTING_STACK_INDEPENDENT_FLAG: bool = False
RECURRENT_HALTING_STACK_HIDDEN_DIM: int | None = None
RECURRENT_HALTING_OUTPUT_DIM: int = HALTING_OUTPUT_DIM
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
# WEIGHT OPTIONS
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
# Weight generator stack options.
# If False, weight generator stack options inherit ADAPTIVE_SUBMODULE_STACK_*.
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
# BIAS OPTIONS
# If `BIAS_OPTION_FLAG` is False, the bias-specific parameters below are ignored.
BIAS_OPTION_FLAG: bool = False
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)
# Bias generator stack options.
# If False, bias generator stack options inherit ADAPTIVE_SUBMODULE_STACK_*.
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
# DIAGONAL OPTIONS
# If `DIAGONAL_OPTION_FLAG` is False, the diagonal-specific parameters below are
# ignored.
DIAGONAL_OPTION_FLAG: bool = False
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None
# Diagonal generator stack options.
# If False, diagonal generator stack options inherit ADAPTIVE_SUBMODULE_STACK_*.
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
# If `MASK_OPTION_FLAG` is False, the mask-specific parameters below are ignored.
MASK_OPTION_FLAG: bool = False
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1
MASK_SURROGATE_SCALE: float = 10.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN
# Mask generator stack options.
# If False, mask generator stack options inherit ADAPTIVE_SUBMODULE_STACK_*.
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
# If False, use the default linear input boundary projector.
INPUT_LAYER_ADAPTIVE_FLAG: bool = False
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
# Input boundary adaptive generator stack options.
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = ADAPTIVE_SUBMODULE_STACK_HIDDEN_DIM
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    ADAPTIVE_SUBMODULE_STACK_LAYER_NORM_POSITION
)
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = ADAPTIVE_SUBMODULE_STACK_NUM_LAYERS
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = (
    ADAPTIVE_SUBMODULE_STACK_ACTIVATION
)
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions
) = ADAPTIVE_SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = (
    ADAPTIVE_SUBMODULE_STACK_DROPOUT_PROBABILITY
)
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    ADAPTIVE_SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
INPUT_LAYER_ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = (
    ADAPTIVE_SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)

#########################################################################
# OUTPUT BOUNDARY PROJECTOR OPTIONS
# If False, use the default linear output boundary projector.
OUTPUT_LAYER_ADAPTIVE_FLAG: bool = False
# Output boundary dynamic weight options.
OUTPUT_LAYER_WEIGHT_OPTION: type[DynamicWeightConfig] | None = None
OUTPUT_LAYER_WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = WEIGHT_GENERATOR_DEPTH
OUTPUT_LAYER_WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WEIGHT_DECAY_SCHEDULE
OUTPUT_LAYER_WEIGHT_DECAY_RATE: float = WEIGHT_DECAY_RATE
OUTPUT_LAYER_WEIGHT_DECAY_WARMUP_BATCHES: int = WEIGHT_DECAY_WARMUP_BATCHES
OUTPUT_LAYER_WEIGHT_NORMALIZATION_OPTION: WeightNormalizationOptions = (
    WEIGHT_NORMALIZATION_OPTION
)
OUTPUT_LAYER_WEIGHT_NORMALIZATION_POSITION_OPTION: (
    WeightNormalizationPositionOptions
) = WEIGHT_NORMALIZATION_POSITION_OPTION
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
# Output boundary adaptive generator stack options.
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = ADAPTIVE_SUBMODULE_STACK_HIDDEN_DIM
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    ADAPTIVE_SUBMODULE_STACK_LAYER_NORM_POSITION
)
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = ADAPTIVE_SUBMODULE_STACK_NUM_LAYERS
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = (
    ADAPTIVE_SUBMODULE_STACK_ACTIVATION
)
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_RESIDUAL_CONNECTION_OPTION: (
    ResidualConnectionOptions
) = ADAPTIVE_SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = (
    ADAPTIVE_SUBMODULE_STACK_DROPOUT_PROBABILITY
)
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    ADAPTIVE_SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
)
OUTPUT_LAYER_ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = (
    ADAPTIVE_SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
)

#########################################################################
# HYPERPARAMETER SEARCH SPACE
# These values define the parameter ranges explored when search mode is enabled.

#########################################################################
# GLOBAL TRAINING AND MAIN LAYER STACK HYPERPARAMETERS
# Global training hyperparameters.
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
# Main stack shape and regularization hyperparameters.
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Main stack normalization and activation hyperparameters.
SEARCH_SPACE_STACK_LAYER_NORM_POSITION: list = [
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
# BOUNDARY PROJECTOR HYPERPARAMETERS
SEARCH_SPACE_INPUT_LAYER_ADAPTIVE_FLAG: list = [False, True]
SEARCH_SPACE_OUTPUT_LAYER_ADAPTIVE_FLAG: list = [False, True]

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
        name="halting",
        label="Halting (adaptive compute)",
        description=(
            "Logs recurrence depth, halting fraction, max-steps saturation, ponder "
            "loss, plus survival heatmap and ponder-cost histogram for "
            "stick-breaking / soft halting modules."
        ),
        kinds=["scalar", "histogram", "image"],
        callback_factory=lambda: HaltingMonitorCallback(log_every_n_steps=100),
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
]
