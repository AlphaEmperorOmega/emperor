from emperor.base.enums import BaseOptions
from emperor.adaptive.utils.layers import AdaptiveParameterLayer
from emperor.adaptive.utils.stack import AdaptiveParameterLayerStack


class AdaptiveParameterLayerOptions(BaseOptions):
    VECTOR = 0
    MATRIX = 1
    GENERATOR = 2


class AdaptiveLayerOptions(BaseOptions):
    BASE = AdaptiveParameterLayer


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = AdaptiveParameterLayerStack
