from Emperor.base.enums import BaseOptions
from Emperor.layers.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.layers.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
)


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    DYNAMIC = DynamicDiagonalLinearLayer


class ParameterGeneratorOptions(BaseOptions):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
