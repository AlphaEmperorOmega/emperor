import torch.nn as nn
from torch import Tensor

from emperor.base.module import Module
from emperor.convs.core._validator import Conv2dLayerValidator
from emperor.convs.core.config import Conv2dLayerConfig


class Conv2dLayer(Module):
    VALIDATOR = Conv2dLayerValidator

    def __init__(
        self,
        cfg: "Conv2dLayerConfig",
        overrides: "Conv2dLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg: Conv2dLayerConfig = self._override_config(cfg, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.kernel_size: int = self.cfg.kernel_size
        self.stride: int = self.cfg.stride
        self.padding: int = self.cfg.padding
        self.bias_flag: bool = self.cfg.bias_flag
        self.VALIDATOR.validate(self)

        self.model = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias_flag,
        )

    def forward(self, X: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(X)
        return self.model(X)
