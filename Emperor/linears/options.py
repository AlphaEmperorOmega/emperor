from Emperor.base.enums import BaseOptions
from Emperor.linears.utils.layers import (
    DynamicLinearLayer,
    LinearLayer,
)


class LinearLayerOptions(BaseOptions):
    BASE = LinearLayer
    DYNAMIC = DynamicLinearLayer
