from models.trainer_config import *
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.halting.options import HaltingHiddenStateModeOptions
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
NUM_EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
TRAINER_ACCELERATOR: str = "cpu"
TRAINER_GRADIENT_CLIP_VAL: float = 1.0
CALLBACK_EARLY_STOPPING_PATIENCE: int = 5

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
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.2
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True

#########################################################################
# GATE STACK OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_HIDDEN_DIM: int = HIDDEN_DIM
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = STACK_LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = 2
GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.TANH
GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
GATE_STACK_DROPOUT_PROBABILITY: float = 0.0
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = STACK_LAST_LAYER_BIAS_OPTION
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = True
GATE_BIAS_FLAG: bool = True

#########################################################################
# HALTING OPTIONS
# If `HALTING_FLAG` is False, the halting-specific parameters below are ignored.
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
# WEIGHT OPTIONS
# If `WEIGHT_OPTION` is None, the weight-specific parameters below are ignored.
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
# BIAS OPTIONS
# If `BIAS_OPTION` is None, the bias-specific parameters below are ignored.
BIAS_OPTION: type[DynamicBiasConfig] | None = None
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)

#########################################################################
# DIAGONAL OPTIONS
# If `DIAGONAL_OPTION` is None, the diagonal-specific parameters below are ignored.
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = None

#########################################################################
# MASK OPTIONS
# If `ROW_MASK_OPTION` is None, the mask-specific parameters below are ignored.
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_THRESHOLD: float = 0.5
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1
MASK_SURROGATE_SCALE: float = 10.0
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN

#########################################################################
# Augmentation generator stack
ADAPTIVE_STACK_HIDDEN_DIM: int = HIDDEN_DIM
ADAPTIVE_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = STACK_LAYER_NORM_POSITION
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
# These values define the parameter ranges explored when search mode is enabled.

#########################################################################
# GLOBAL TRAINING AND MAIN LAYER STACK HYPERPARAMETERS
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [16, 32, 64, 128, 256, 512]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [2, 4, 8, 16, 32]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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
