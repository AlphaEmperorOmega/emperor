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
    WEIGHTED_BANK = 5


class DynamicWeightOptions(Enum):
    DISABLED = 0
    DUAL_MODEL = 1
    SINGLE_MODEL = 2
    LOW_RANK = 3
    HYPERNETWORK = 4
    LAYERED_WEIGHTED_BANK = 5
    SOFT_WEIGHTED_BANK = 6


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
    GLOBAL_SCORE = 1
    PER_ROW_SCORE = 2
    TOP_SLICE = 3
    DIAGONAL = 4


class MaskDimensionOptions(Enum):
    ROW = "mask_rows"
    COLUMN = "mask_columns"


class LinearMemoryPositionOptions(Enum):
    BEFORE_AFFINE = 1
    AFTER_AFFINE = 2


class WeightNormalizationPositionOptions(Enum):
    DISABLED = 0
    BEFORE_OUTER_PRODUCT = 1
    AFTER_OUTER_PRODUCT = 2


class WeightDecayScheduleOptions(Enum):
    DISABLED = 0
    EXPONENTIAL = 1
    LINEAR = 2
    MULTIPLICATIVE = 3
