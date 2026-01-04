from Emperor.attention.utils.layer import MultiHeadAttention
from Emperor.base.enums import BaseOptions


class AdaptiveLayerOptions(BaseOptions):
    BASE = MultiHeadAttention


class AdaptiveLayerStackOptions(BaseOptions):
    BASE = MultiHeadAttention
