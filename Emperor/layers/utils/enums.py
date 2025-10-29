from enum import Enum
from Emperor.layers.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.layers.utils.linears import (
    DynamicLinearLayer,
    LinearLayer,
)


class LinearLayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC = DynamicLinearLayer


class ParameterGeneratorTypes(Enum):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer


class LayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC_BASE = DynamicLinearLayer
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
