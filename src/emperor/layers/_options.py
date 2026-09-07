from collections.abc import Callable
from enum import member
from typing import cast

import torch.nn.functional as F
from torch import Tensor

from emperor.config import BaseOptions


class ActivationOptions(BaseOptions):
    def __call__(self, x: Tensor) -> Tensor:
        activation_function = cast(Callable[[Tensor], Tensor], self.value)
        return activation_function(x)

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


class NormalizationOptions(BaseOptions):
    """Last-dimension normalization and learned elementwise replacements."""

    RMS_NORM = 1
    LAYER_NORM = 2
    DYNAMIC_TANH = 3
    DERF = 4
    DYISRU = 5


class LayerNormPositionOptions(BaseOptions):
    DISABLED = 0
    DEFAULT = 1
    BEFORE = 2
    AFTER = 3


class LastLayerBiasOptions(BaseOptions):
    DEFAULT = 0
    DISABLED = 1
    ENABLED = 2


class LayerGateOptions(BaseOptions):
    MULTIPLIER = 1
    ADDITION = 2
