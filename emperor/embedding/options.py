from emperor.base.options import BaseOptions


class RelativePositionalEmbeddingOptions(BaseOptions):
    DISABLED = 0
    LEARNED = 1


class AbsolutePositionalEmbeddingOptions(BaseOptions):
    DISABLED = 0
    SINUSOIDAL = 1
    LEARNED = 2
