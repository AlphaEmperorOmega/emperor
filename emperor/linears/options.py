from emperor.base.enums import BaseOptions
from emperor.linears.core.layers import AdaptiveLinearLayer, LinearLayer
from emperor.linears.core.stack import AdaptiveLinearLayerStack, LinearLayerStack


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    ADAPTIVE = AdaptiveLinearLayer


class LinearLayerStackOptions(BaseOptions):
    BASE = LinearLayerStack
    ADAPTIVE = AdaptiveLinearLayerStack
