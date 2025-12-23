from Emperor.base.enums import BaseOptions


class AdaptiveWeightOptions(BaseOptions):
    VECTOR = 1
    MATRIX = 2
    GENERATOR = 3


class AdaptiveBiasOptions(BaseOptions):
    DISABLED = 0
    MATRIX = 1
    GENERATOR = 2
