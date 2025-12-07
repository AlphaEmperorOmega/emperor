from Emperor.base.enums import BaseOptions
from Emperor.adaptive.utils.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)


class AdaptiveLayerOptions(BaseOptions):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
