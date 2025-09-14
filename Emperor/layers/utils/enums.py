from enum import Enum
from Emperor.layers.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.layers.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)


class LinearLayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC = DynamicDiagonalLinearLayer


class ParameterGeneratorTypes(Enum):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer


class LayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC_BASE = DynamicDiagonalLinearLayer
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
