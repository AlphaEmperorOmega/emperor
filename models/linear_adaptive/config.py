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
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
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
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
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
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
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
HALTING_GATE_HIDDEN_DIM: int = HIDDEN_DIM
HALTING_GATE_OUTPUT_DIM: int = 2
HALTING_GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LayerNormPositionOptions.DISABLED
)
HALTING_GATE_STACK_NUM_LAYERS: int = 2
HALTING_GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
HALTING_GATE_STACK_RESIDUAL_FLAG: bool = False
HALTING_GATE_STACK_DROPOUT_PROBABILITY: float = 0.0
HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_GATE_BIAS_FLAG: bool = BIAS_FLAG

#########################################################################
# WEIGHT OPTIONS
# If `WEIGHT_FLAG` is False, the weight-specific parameters below are ignored.
WEIGHT_FLAG: bool = False
WEIGHT_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
WEIGHT_DECAY_RATE: float = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: int = 0
WEIGHT_NORMALIZATION: WeightNormalizationOptions = WeightNormalizationOptions.DISABLED
WEIGHT_NORMALIZATION_POSITION: WeightNormalizationPositionOptions = (
    WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
)
WEIGHT_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_THREE
)


#########################################################################
# BIAS OPTIONS
# If `BIAS_OPTION` is None, the bias-specific parameters below are ignored.
BIAS_OPTION: type[DynamicBiasConfig] | None = GeneratorDynamicBiasConfig
BIAS_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)

#########################################################################
# DIAGONAL OPTIONS
# If `DIAGONAL_OPTION` is None, the diagonal-specific parameters below are ignored.
DIAGONAL_OPTION: type[DynamicDiagonalConfig] | None = CombinedDynamicDiagonalConfig
DIAGONAL_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE

#########################################################################
# MASK OPTIONS
# If `ROW_MASK_OPTION` is None, the mask-specific parameters below are ignored.
ROW_MASK_OPTION: type[AxisMaskConfig] | None = None
MASK_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN
MASK_THRESHOLD: float = 0.5
MASK_SURROGATE_SCALE: float = 10.0
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1

#########################################################################
# Augmentation generator stack
AUGMENTATION_GENERATOR_STACK_HIDDEN_DIM: int = HIDDEN_DIM
AUGMENTATION_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LAYER_NORM_POSITION
)
AUGMENTATION_GENERATOR_STACK_NUM_LAYERS: int = 2
AUGMENTATION_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
AUGMENTATION_GENERATOR_STACK_RESIDUAL_FLAG: bool = False
AUGMENTATION_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.1
AUGMENTATION_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
AUGMENTATION_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
# HYPERPARAMETER SEARCH SPACE
# These values define the parameter ranges explored when search mode is enabled.
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
SEARCH_SPACE_WEIGHT_GENERATOR_DEPTH: list = [
    DynamicDepthOptions.DEPTH_OF_ONE,
    DynamicDepthOptions.DEPTH_OF_TWO,
    DynamicDepthOptions.DEPTH_OF_THREE,
    DynamicDepthOptions.DEPTH_OF_FOUR,
    DynamicDepthOptions.DEPTH_OF_FIVE,
    DynamicDepthOptions.DEPTH_OF_SIX,
    DynamicDepthOptions.DEPTH_OF_SEVEN,
    DynamicDepthOptions.DEPTH_OF_EIGHT,
    DynamicDepthOptions.DEPTH_OF_NINE,
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
SEARCH_SPACE_WEIGHT_NORMALIZATION: list = [
    WeightNormalizationOptions.DISABLED,
    WeightNormalizationOptions.CLAMP,
    WeightNormalizationOptions.L2_SCALE,
    WeightNormalizationOptions.SOFT_CLAMP,
    WeightNormalizationOptions.RMS,
    WeightNormalizationOptions.SIGMOID_SCALE,
]
SEARCH_SPACE_WEIGHT_NORMALIZATION_POSITION: list = [
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
SEARCH_SPACE_BIAS_OPTION: list = [
    None,
    AffineTransformDynamicBiasConfig,
    AdditiveDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
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
