import math

import torch
import torch.nn as nn

from torch import Tensor
from emperor.base.utils import Module
from emperor.base.options import BaseOptions


class ResidualConnectionOptions(BaseOptions):
    DISABLED = 0
    RESIDUAL = 1
    WEIGHTED_RESIDUAL = 2
    WEIGHTED_BLEND = 3


class ResidualConnection(Module):
    WEIGHTED_BLEND_INITIAL_ALPHA = 0.9

    def __init__(
        self,
        option: ResidualConnectionOptions,
    ):
        super().__init__()
        self.option = option
        self.raw_weight = self.__init_raw_weight()

    def __init_raw_weight(self) -> nn.Parameter | None:
        if self.option == ResidualConnectionOptions.WEIGHTED_RESIDUAL:
            raw_mix_coefficient = torch.tensor(0.0)
            return nn.Parameter(raw_mix_coefficient)
        if self.option == ResidualConnectionOptions.WEIGHTED_BLEND:
            alpha = self.WEIGHTED_BLEND_INITIAL_ALPHA
            raw_alpha = torch.tensor(math.log(alpha / (1.0 - alpha)))
            return nn.Parameter(raw_alpha)
        return None

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        if self.option == ResidualConnectionOptions.DISABLED:
            return current
        if self.option == ResidualConnectionOptions.RESIDUAL:
            return current + previous
        if self.option == ResidualConnectionOptions.WEIGHTED_RESIDUAL:
            residual_weight = torch.tanh(self.raw_weight)
            return previous + residual_weight * current
        if self.option == ResidualConnectionOptions.WEIGHTED_BLEND:
            alpha = torch.sigmoid(self.raw_weight)
            return alpha * current + (1.0 - alpha) * previous
        raise ValueError(
            f"Unsupported residual connection option {self.option} for ResidualConnection."
        )
