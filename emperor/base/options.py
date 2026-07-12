from enum import member

import torch.nn.functional as F

from emperor.base.option import BaseOptions


class ActivationOptions(BaseOptions):
    def __call__(self, x):
        return self.value(x)

    DISABLED = 0
    RELU = member(F.relu)
    GELU = member(F.gelu)
    SIGMOID = member(F.sigmoid)
    TANH = member(F.tanh)
    LEAKY_RELU = member(F.leaky_relu)
    ELU = member(F.elu)
    SELU = member(F.selu)
    SOFTPLUS = member(F.softplus)
    SOFTSIGN = member(F.softsign)
    SILU = member(F.silu)
    MISH = member(F.mish)


class LayerNormPositionOptions(BaseOptions):
    DISABLED = 0
    DEFAULT = 1
    BEFORE = 2
    AFTER = 3


class LastLayerBiasOptions(BaseOptions):
    DEFAULT = 0
    DISABLED = 1
    ENABLED = 2
