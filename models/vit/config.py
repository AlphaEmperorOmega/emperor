from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.base.enums import ActivationOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.datasets.image.classification.cifar_10 import Cifar10
from emperor.datasets.image.classification.cifar_100 import Cifar100
from emperor.datasets.image.classification.fashion_mnist import FashionMNIST
from emperor.transformer.utils.layers import TransformerConfig
from emperor.transformer.utils.patch.options.base import PatchConfig
from emperor.embedding.absolute.config import AbsolutePositionalEmbeddingConfig
from emperor.linears.options import LinearLayerStackOptions
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)
from models.trainer_config import *

# Global
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
SEARCH_SPACE_LEARNING_RATE: list = [1e-4, 1e-3, 1e-2]

# Trainer
# ACCELERATOR: str = "cpu"
GRADIENT_CLIP_VAL: float = 1.0
EARLY_STOPPING_PATIENCE: int = 5

# Model
INPUT_DIM: int = 32
OUTPUT_DIM: int = 10
OUTPUT_BIAS_FLAG: bool = True
IMAGE_PATCH_SIZE: int = 4
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 32

# Shared
HIDDEN_DIM: int = 32
DROPOUT_PROBABILITY: float = 0.1
ACTIVATION_FUNCTION: ActivationOptions = ActivationOptions.SILU
TRANSFORMER_NUM_LAYERS: int = 1
ATTN_NUM_LAYERS: int = 1
FF_BIAS_FLAG: bool = True
FF_NUM_LAYERS: int = 2

# Preset
ATTN_BIAS_FLAG: bool = False
ATTN_NUM_HEADS: int = 4
ATTN_MODEL_TYPE: LinearLayerStackOptions = LinearLayerStackOptions.BASE
FF_MODEL_TYPE: LinearLayerStackOptions = LinearLayerStackOptions.BASE
OUTPUT_NUM_LAYERS: int = 2

# Adaptive preset
ADAPTIVE_ATTN_BIAS_FLAG: bool = True
ADAPTIVE_ATTN_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO
ADAPTIVE_ATTN_DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL
ADAPTIVE_ATTN_BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DISABLED
ADAPTIVE_ATTN_BEHAVIOUR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_FF_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO
ADAPTIVE_FF_DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL
ADAPTIVE_FF_BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DISABLED
ADAPTIVE_FF_BEHAVIOUR_STACK_NUM_LAYERS: int = 2
ADAPTIVE_OUTPUT_NUM_LAYERS: int = 1
ADAPTIVE_OUTPUT_GENERATOR_DEPTH: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO
ADAPTIVE_OUTPUT_DIAGONAL_OPTION: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL
ADAPTIVE_OUTPUT_BIAS_OPTION: DynamicBiasOptions = DynamicBiasOptions.DISABLED
ADAPTIVE_OUTPUT_BEHAVIOUR_STACK_NUM_LAYERS: int = 2


@dataclass
class ExperimentConfig(ConfigBase):
    patch_config: "PatchConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    encoder_config: "TransformerConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    output_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
