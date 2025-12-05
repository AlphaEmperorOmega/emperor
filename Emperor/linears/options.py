from Emperor.base.enums import BaseOptions
from Emperor.linears.utils.layers import AdaptiveLinearLayer, LinearLayer
from Emperor.linears.utils.stack import AdaptiveLinearLayerStack, LinearLayerStack


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    ADAPTIVE = AdaptiveLinearLayer


class LinearLayerStackOptions(BaseOptions):
    BASE = LinearLayerStack
    ADAPTIVE = AdaptiveLinearLayerStack
