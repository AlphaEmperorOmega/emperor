from enum import Enum
import torch.nn.functional as F

from Emperor.attention.attention import MultiHeadAttention
from Emperor.layers.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.layers.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)


class LayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC_BASE = DynamicDiagonalLinearLayer
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer


class LinearLayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC_BASE = DynamicDiagonalLinearLayer


class ParameterGeneratorTypes(Enum):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer


class FeedForwardTypes(Enum):
    BASE = "TO BE IMPLEMENTED"


class AttentionTypes(Enum):
    BASE = MultiHeadAttention


class ActivationOptions(Enum):
    RELU = F.relu
    GELU = F.gelu
    SIGMOID = F.sigmoid
    TANH = F.tanh
    LEAKY_RELU = F.leaky_relu
    ELU = F.elu
    SELU = F.selu
    SOFTPLUS = F.softplus
    SOFTSIGN = F.softsign
