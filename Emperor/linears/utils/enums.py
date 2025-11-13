from enum import Enum


class OuterProductNormOptions(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    LAYER_NORM = 4


class DynamicDiagonalOptions(Enum):
    DEFAULT = 0
    DIAGONAL = 1
    ANTI_DIAGONAL = 2
    DIAGONAL_AND_ANTI_DIAGONAL = 3


class DynamicBiasOptions(Enum):
    DEFAULT = 1
    SCALE_AND_OFFSET = 2
    ELEMENT_WISE_OFFSET = 3
    DYNAMIC_PARAMETERS = 4


class DynamicParametersOptions(Enum):
    DEFAULT = 0
    ONE_OPINION = 1
    TWO_OPINIONS = 2
    THREE_OPINIONS = 3
