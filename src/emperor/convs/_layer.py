import torch.nn as nn
from torch import Tensor

from emperor.convs._config import Conv2dLayerConfig
from emperor.convs._validation import Conv2dLayerValidator
from emperor.nn import Module


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

        self.model = self.__build_configured_conv2d_layer()

    def __build_configured_conv2d_layer(self) -> nn.Conv2d:
        return nn.Conv2d(
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
