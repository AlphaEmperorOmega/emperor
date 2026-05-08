from enum import Enum


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
    DEPTH_OF_FOUR = 4
    DEPTH_OF_FIVE = 5
    DEPTH_OF_SIX = 6
    DEPTH_OF_SEVEN = 7
    DEPTH_OF_EIGHT = 8
    DEPTH_OF_NINE = 9
    DEPTH_OF_TEN = 10


class BankExpansionFactorOptions(Enum):
    DISABLED = 0
    FACTOR_OF_ONE = 1
    FACTOR_OF_TWO = 2
    FACTOR_OF_THREE = 3
    FACTOR_OF_FOUR = 4


class MaskDimensionOptions(Enum):
    ROW = 0
    COLUMN = 1


class WeightNormalizationPositionOptions(Enum):
    DISABLED = 0
    BEFORE_OUTER_PRODUCT = 1
    AFTER_OUTER_PRODUCT = 2


class WeightDecayScheduleOptions(Enum):
    DISABLED = 0
    EXPONENTIAL = 1
    LINEAR = 2
    MULTIPLICATIVE = 3
