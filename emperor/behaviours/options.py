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
    GATED = 4


class DynamicWeightOptions(Enum):
    DISABLED = 0
    DUAL_MODEL = 1
    SINGLE_MODEL = 2
    LOW_RANK = 3
    WEIGHT_MASK = 4
    HYPERNETWORK = 5


class WeightNormalizationOptions(Enum):
    DISABLED = 0
    CLAMP = 1
    L2_SCALE = 2
    SOFT_CLAMP = 3
    RMS = 4
    SIGMOID_SCALE = 5


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


class RowMaskOptions(Enum):
    DISABLED = 0
    ENABLED = 1


class LinearMemoryPositionOptions(Enum):
    BEFORE_AFFINE = 1
    AFTER_AFFINE = 2
