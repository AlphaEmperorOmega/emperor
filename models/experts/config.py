from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.enums import ActivationOptions
from emperor.experts.utils.enums import DroppedTokenOptions, ExpertWeightingPositionOptions, InitSamplerOptions
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

# Trainer
# ACCELERATOR: str = "cpu"
GRADIENT_CLIP_VAL: float = 1.0
EARLY_STOPPING_PATIENCE: int = 5

# Model
INPUT_DIM: int = 28**2
OUTPUT_DIM: int = 10
BIAS_FLAG: bool = True

# Output layer stack
OUTPUT_NUM_LAYERS: int = 2
OUTPUT_ACTIVATION: ActivationOptions = ActivationOptions.SILU
OUTPUT_DROPOUT_PROBABILITY: float = 0.1

# Sampler
ROUTER_NOISY_TOPK_FLAG: bool = False
SAMPLER_THRESHOLD: float = 0.0
SAMPLER_FILTER_ABOVE_THRESHOLD: bool = False
SAMPLER_NUM_TOPK_SAMPLES: int = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG: bool = False
SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SAMPLER_SWITCH_LOSS_WEIGHT: float = 0.0
SAMPLER_ZERO_CENTRED_LOSS_WEIGHT: float = 0.0
SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

# Experts
EXPERTS_TOP_K: int = 3
EXPERTS_NUM_EXPERTS: int = 6
EXPERTS_COMPUTE_EXPERT_MIXTURE_FLAG: bool = False
EXPERTS_WEIGHTED_PARAMETERS_FLAG: bool = False
EXPERTS_WEIGHTING_POSITION_OPTION: ExpertWeightingPositionOptions = ExpertWeightingPositionOptions.BEFORE_EXPERTS
EXPERTS_INIT_SAMPLER_OPTION: InitSamplerOptions = InitSamplerOptions.DISABLED
EXPERTS_CAPACITY_FACTOR: float = 0.0
EXPERTS_DROPPED_TOKEN_BEHAVIOR: DroppedTokenOptions = DroppedTokenOptions.ZEROS
EXPERTS_MODEL_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DISABLED
EXPERTS_MODEL_DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DISABLED
EXPERTS_MODEL_BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DISABLED
EXPERTS_MODEL_MEMORY_OPTION: LinearMemoryOptions = LinearMemoryOptions.DISABLED
EXPERTS_MODEL_MEMORY_SIZE_OPTION: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED
EXPERTS_MODEL_MEMORY_POSITION_OPTION: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE

# Layer stack options
HIDDEN_DIM: int = 256
STACK_NUM_LAYERS: int = 3
STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.0

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
SEARCH_SPACE_EXPERTS_TOP_K: list = [1, 2, 3]
SEARCH_SPACE_EXPERTS_NUM_EXPERTS: list = [4, 6, 8]


@dataclass
class ExperimentConfig(ConfigBase):
    experts_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
