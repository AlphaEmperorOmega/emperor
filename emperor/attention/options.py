from emperor.attention.utils.layer import MultiHeadAttention
from emperor.base.enums import BaseOptions


class AdaptiveLayerOptions(BaseOptions):
    BASE = MultiHeadAttention


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = MultiHeadAttention
