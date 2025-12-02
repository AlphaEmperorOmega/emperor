from Emperor.base.enums import BaseOptions
from Emperor.linears.utils.layers import AdaptiveLinearLayer, LinearLayer


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    DYNAMIC = AdaptiveLinearLayer
