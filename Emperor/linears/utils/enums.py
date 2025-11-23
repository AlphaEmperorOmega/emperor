from enum import Enum


class DynamicDiagonalOptions(Enum):
    DISABLED = 0
    DIAGONAL = 1
    ANTI_DIAGONAL = 2
    DIAGONAL_AND_ANTI_DIAGONAL = 3


class DynamicBiasOptions(Enum):
    DISABLED = 0
    SCALE_AND_OFFSET = 1
    ELEMENT_WISE_OFFSET = 2
    DYNAMIC_PARAMETERS = 3


class DynamicDepthOptions(Enum):
    DISABLED = 0
    DEPTH_OF_ONE = 1
    DEPTH_OF_TWO = 2
    DEPTH_OF_THREE = 3


class LinearMemorySizeOptions(Enum):
    DISABLED = 0
    SMALL = 4
    MEDIUM = 8
    LARGE = 16
    MAX = 32


class LinearMemoryOptions(Enum):
    DISABLED = 0
    FUSION = 1
    WEIGHTED = 2


class LinearMemoryPositionOptions(Enum):
    BEFORE_AFFINE = 1
    AFTER_AFFINE = 2
