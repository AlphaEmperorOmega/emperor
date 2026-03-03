from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

BATCH_SIZE: int = 64
INPUT_DIM: int = 28**2
HIDDEN_DIM: int = 256
OUTPUT_DIM: int = 10
BIAS_FLAG: bool = True
LAYER_NORM_POSITION: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT
GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_THREE
DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL
BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DYNAMIC_PARAMETERS
MEMORY_OPTION: LinearMemoryOptions = LinearMemoryOptions.DISABLED
MEMORY_SIZE_OPTION: LinearMemorySizeOptions = LinearMemorySizeOptions.DISABLED
MEMORY_POSITION_OPTION: LinearMemoryPositionOptions = LinearMemoryPositionOptions.BEFORE_AFFINE
STACK_NUM_LAYERS: int = 3
STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
STACK_RESIDUAL_FLAG: bool = False
STACK_DROPOUT_PROBABILITY: float = 0.0
ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: int = 256
ADAPTIVE_GENERATOR_STACK_ACTIVATION: ActivationOptions = ActivationOptions.RELU
ADAPTIVE_GENERATOR_STACK_RESIDUAL_FLAG: bool = False
ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: float = 0.0

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
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_NUM_LAYERS: list = [1, 2, 3]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_HIDDEN_DIM: list = [64, 128, 256]
SEARCH_SPACE_ADAPTIVE_GENERATOR_STACK_DROPOUT_PROBABILITY: list = [0.0, 0.1]


@dataclass
class ExperimentConfig(ConfigBase):
    model_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
