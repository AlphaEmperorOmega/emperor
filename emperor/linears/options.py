from emperor.base.enums import BaseOptions
from emperor.linears.utils.layers import AdaptiveLinearLayer, LinearLayer
from emperor.linears.utils.stack import AdaptiveLinearLayerStack, LinearLayerStack


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    ADAPTIVE = AdaptiveLinearLayer


class LinearLayerStackOptions(BaseOptions):
    BASE = LinearLayerStack
    ADAPTIVE = AdaptiveLinearLayerStack
