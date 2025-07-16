from enum import Enum
import torch.nn.functional as F

from Emperor.components.parameter_generators.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.components.parameter_generators.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)


class LayerTypes(Enum):
    BASE: LinearLayer
    DYNAMIC_BASE: DynamicDiagonalLinearLayer
    VECTOR: VectorParameterLayer
    MATRIX: MatrixParameterLayer
    GENERATOR: GeneratorParameterLayer


class ActivationFunctionOptions(Enum):
    RELU = F.relu
    GELU = F.gelu
    SIGMOID = F.sigmoid
    TANH = F.tanh
    LEAKY_RELU = F.leaky_relu
    ELU = F.elu
    SELU = F.selu
    SOFTPLUS = F.softplus
    SOFTSIGN = F.softsign
