from emperor.base.enums import BaseOptions
from emperor.parametric.utils.layers import ParametricLayer
from emperor.parametric.utils.stack import ParametricLayerStack


class ParametricLayerOptions(BaseOptions):
    VECTOR = 0
    MATRIX = 1
    GENERATOR = 2


class AdaptiveLayerOptions(BaseOptions):
    BASE = ParametricLayer


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = ParametricLayerStack
