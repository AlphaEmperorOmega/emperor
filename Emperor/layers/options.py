from Emperor.base.enums import BaseOptions
from Emperor.layers.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)
from Emperor.layers.utils.linears import (
    DynamicLinearLayer,
    LinearLayer,
)


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    DYNAMIC = DynamicLinearLayer


class ParameterGeneratorOptions(BaseOptions):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
