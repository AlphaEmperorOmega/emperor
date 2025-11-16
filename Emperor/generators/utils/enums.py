from enum import Enum
from Emperor.linears.utils.handlers.parameter import DepthMappingLayer
from Emperor.generators.utils.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.generators.utils.linears import (
    DynamicLinearLayer,
    LinearLayer,
)


class LinearLayerTypes(Enum):
    BASE = LinearLayer
    DYNAMIC = DynamicLinearLayer
    DYNAMIC_PARAMETERS = DepthMappingLayer


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
