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
    AxisMaskOptions,
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
    WeightDecayScheduleOptions,
    MaskDimensionOptions,
    BankExpansionFactorOptions,
)

# Global
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 2
LEARNING_RATE: float = 1e-3
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Trainer
# ACCELERATOR: str = "cpu"
GRADIENT_CLIP_VAL: float = 1.0
EARLY_STOPPING_PATIENCE: int = 5

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10
BIAS_FLAG: bool = True

# Layer stack options
HIDDEN_DIM: int = 256
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.BEFORE
STACK_NUM_LAYERS: int = 3
STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.1
STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT
STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

#########################################################################
# GATE STACK OPTIONS
# If `GATE_FLAG` is False, the gate-specific parameters below are ignored.
GATE_FLAG: bool = False
GATE_HIDDEN_DIM: int = HIDDEN_DIM
GATE_LAYER_NORM_POSITION: LayerNormPositionOptions = LAYER_NORM_POSITION
GATE_STACK_NUM_LAYERS: int = STACK_NUM_LAYERS
GATE_STACK_ACTIVATION: ActivationOptions = STACK_ACTIVATION
GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
GATE_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = STACK_LAST_LAYER_BIAS_OPTION
GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
GATE_BIAS_FLAG: bool = BIAS_FLAG

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
HALTING_GATE_STACK_NUM_LAYERS: int = STACK_NUM_LAYERS
HALTING_GATE_STACK_ACTIVATION: ActivationOptions = ActivationOptions.DISABLED
HALTING_GATE_STACK_RESIDUAL_FLAG: bool = STACK_RESIDUAL_FLAG
HALTING_GATE_STACK_DROPOUT_PROBABILITY: float = STACK_DROPOUT_PROBABILITY
HALTING_GATE_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DISABLED
)
HALTING_GATE_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False
HALTING_GATE_BIAS_FLAG: bool = True

# Adaptive behaviour
GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
DIAGONAL_OPTION: DynamicDiagonalOptions = (
    DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL
)
BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DYNAMIC_PARAMETERS
WEIGHT_FLAG: bool = False
WEIGHT_NORMALIZATION: WeightNormalizationOptions = WeightNormalizationOptions.DISABLED
WEIGHT_NORMALIZATION_POSITION: WeightNormalizationPositionOptions = (
    WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT
)
WEIGHT_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
WEIGHT_DECAY_RATE: float = 0.0
WEIGHT_DECAY_WARMUP_BATCHES: int = 0
BIAS_DECAY_SCHEDULE: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED
BIAS_DECAY_RATE: float = 0.0
BIAS_DECAY_WARMUP_BATCHES: int = 0
BIAS_BANK_EXPANSION_FACTOR: BankExpansionFactorOptions = (
    BankExpansionFactorOptions.FACTOR_OF_TWO
)
ROW_MASK_OPTION: AxisMaskOptions = AxisMaskOptions.DISABLED
MASK_DIMENSION_OPTION: MaskDimensionOptions = MaskDimensionOptions.COLUMN
MASK_THRESHOLD: float = 0.5
MASK_SURROGATE_SCALE: float = 10.0
MASK_FLOOR: float = 0.0
MASK_TRANSITION_WIDTH: float = 0.1

# Adaptive generator stack
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = 256
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.GELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.1
ADAPTIVE_GENERATOR_STACK_LAYER_NORM_POSITION: LayerNormPositionOptions = (
    LAYER_NORM_POSITION
)
ADAPTIVE_GENERATOR_STACK_LAST_LAYER_BIAS_OPTION: LastLayerBiasOptions = (
    LastLayerBiasOptions.DEFAULT
)
ADAPTIVE_GENERATOR_STACK_APPLY_OUTPUT_PIPELINE_FLAG: bool = False

# Hyperparameter search space
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [1, 3, 5, 7]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1, 0.2]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.SILU,
    ActivationOptions.GELU,
    ActivationOptions.MISH,
]
SEARCH_SPACE_STACK_LAYER_NORM_POSITION: list = [
    LayerNormPositionOptions.DISABLED,
    LayerNormPositionOptions.DEFAULT,
    LayerNormPositionOptions.BEFORE,
    LayerNormPositionOptions.AFTER,
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
