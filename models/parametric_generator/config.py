from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.enums import ActivationOptions
from emperor.parametric.utils.mixtures.options import AdaptiveBiasOptions
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from models.trainer_config import *

# Global
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 10
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10

# Layer stack options
HIDDEN_DIM: int = 256
STACK_NUM_LAYERS: int = 3
STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.0

# Adaptive mixture
ADAPTIVE_MIXTURE_TOP_K: int = 3
ADAPTIVE_MIXTURE_NUM_EXPERTS: int = 6
ADAPTIVE_MIXTURE_WEIGHTED_PARAMETERS_FLAG: bool = False
ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION: ClipParameterOptions = ClipParameterOptions.BEFORE
ADAPTIVE_MIXTURE_CLIP_RANGE: float = 5.0
ADAPTIVE_BIAS_OPTION: AdaptiveBiasOptions = AdaptiveBiasOptions.DISABLED

# Adaptive behaviour
ADAPTIVE_BEHAVIOUR_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DISABLED
ADAPTIVE_BEHAVIOUR_DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED
ADAPTIVE_BEHAVIOUR_BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DISABLED
ADAPTIVE_BEHAVIOUR_MEMORY_OPTION: LinearMemoryOptions = LinearMemoryOptions.DISABLED
ADAPTIVE_BEHAVIOUR_MEMORY_SIZE_OPTION: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED
ADAPTIVE_BEHAVIOUR_MEMORY_POSITION_OPTION: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE

# Adaptive generator stack
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = 256
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0

# Hyperparameter search space
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_STACK_NUM_LAYERS: list = [3, 6]
SEARCH_SPACE_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1]
SEARCH_SPACE_STACK_ACTIVATION: list = [
    ActivationOptions.RELU,
    ActivationOptions.SILU,
    ActivationOptions.GELU,
    ActivationOptions.LEAKY_RELU,
]
SEARCH_SPACE_ADAPTIVE_MIXTURE_TOP_K: list = [1, 3]
SEARCH_SPACE_ADAPTIVE_MIXTURE_NUM_EXPERTS: list = [4, 6, 8]
SEARCH_SPACE_ADAPTIVE_BIAS_OPTION: list = [
    AdaptiveBiasOptions.DISABLED,
    AdaptiveBiasOptions.GENERATOR,
]
SEARCH_SPACE_ADAPTIVE_MIXTURE_CLIP_PARAMETER_OPTION: list = [
    ClipParameterOptions.NONE,
    ClipParameterOptions.BEFORE,
    ClipParameterOptions.AFTER,
]
SEARCH_SPACE_ADAPTIVE_BEHAVIOUR_GENERATOR_DEPTH: list = [
    DynamicDepthOptions.DISABLED,
    DynamicDepthOptions.DEPTH_OF_ONE,
    DynamicDepthOptions.DEPTH_OF_TWO,
]


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
