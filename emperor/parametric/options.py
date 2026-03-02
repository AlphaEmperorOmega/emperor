from emperor.base.enums import BaseOptions
from emperor.parametric.utils.layers import AdaptiveParameterLayer
from emperor.parametric.utils.stack import AdaptiveParameterLayerStack


class AdaptiveParameterLayerOptions(BaseOptions):
    VECTOR = 0
    MATRIX = 1
    GENERATOR = 2


class AdaptiveLayerOptions(BaseOptions):
    BASE = AdaptiveParameterLayer


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = AdaptiveParameterLayerStack
