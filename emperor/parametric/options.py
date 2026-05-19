from emperor.base.options import BaseOptions
from emperor.parametric.core.layers import ParametricLayer
from emperor.parametric.core.stack import ParametricLayerStack


class ParametricLayerOptions(BaseOptions):
    VECTOR = 0
    MATRIX = 1
    GENERATOR = 2


class AdaptiveLayerOptions(BaseOptions):
    BASE = ParametricLayer


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = ParametricLayerStack
