from Emperor.base.enums import BaseOptions
from Emperor.generators.utils.layers import (
    GeneratorParameterLayer,
    MatrixParameterLayer,
    VectorParameterLayer,
)


class ParameterGeneratorOptions(BaseOptions):
    VECTOR = VectorParameterLayer
    MATRIX = MatrixParameterLayer
    GENERATOR = GeneratorParameterLayer
