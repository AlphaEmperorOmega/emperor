from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from emperor.linears._validation import LinearValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.linears._config import LinearLayerConfig


class LinearAbstract(Module):
    VALIDATOR = LinearValidator

    def __init__(
        self,
        cfg: "LinearLayerConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg: LinearLayerConfig = self._override_config(cfg, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.VALIDATOR.validate(self)
        self.__init_parameters()

    def __init_parameters(self) -> None:
        self.weight_params = self.__init_weight_parameters()
        self.bias_params = self.__init_bias_parameters()

    def __init_weight_parameters(self) -> Parameter:
        weight_shape = (self.input_dim, self.output_dim)
        return self._init_parameter_bank(weight_shape)

    def __init_bias_parameters(self) -> Parameter | None:
        if not self.bias_flag:
            return None
        bias_shape = (self.output_dim,)
        return self._init_parameter_bank(bias_shape, nn.init.zeros_)


class LinearLayer(LinearAbstract):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

    def forward(self, X: Tensor) -> Tensor:
        self.VALIDATOR.validate_input_tensor(X, self.input_dim)
        return F.linear(X, self.weight_params.T, self.bias_params)
