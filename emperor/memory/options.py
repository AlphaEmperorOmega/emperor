from enum import Enum


class DynamicMemoryOptions(Enum):
    DISABLED = 0
    FUSION = 1
    WEIGHTED = 2
    ELEMENT_WISE_WEIGHTED = 3
    ATTENTION = 4


class MemoryPositionOptions(Enum):
    BEFORE_AFFINE = 1
    AFTER_AFFINE = 2
