import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from emperor.base.utils import Module
from emperor.linears.core._validator import LinearValidator
from emperor.linears.core.config import LinearLayerConfig, AdaptiveLinearLayerConfig
from emperor.augmentations.adaptive_parameters.model import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class LinearBase(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "LinearLayerConfig" = self._override_config(config, overrides)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.bias_flag: bool = self.cfg.bias_flag
        self.__init_parameters()
        LinearValidator.validate(self)

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


class LinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

    def forward(self, X: Tensor) -> Tensor:
        LinearValidator.validate_input_shape(X)
        return F.linear(X, self.weight_params.T, self.bias_params)


class AdaptiveLinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "AdaptiveLinearLayerConfig | ModelConfig",
        overrides: "AdaptiveLinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        LinearValidator.validate_adaptive(self.cfg)
        self.adaptive_augmentation_config = self.cfg.adaptive_augmentation_config
        self.adaptive_behaviour = self.__init_behaviour()

    def __init_behaviour(self):
        overrides = AdaptiveParameterAugmentationConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return AdaptiveParameterAugmentation(
            self.adaptive_augmentation_config, overrides
        )

    def forward(self, X: Tensor) -> Tensor:
        LinearValidator.validate_input_shape(X)
        return self.adaptive_behaviour.compute_adaptive_parameters(
            self._compute_affine_transformation_callback,
            self.weight_params,
            self.bias_params,
            X,
        )

    def _compute_affine_transformation_callback(
        self, weights: Tensor, bias: Tensor | None, X: Tensor
    ) -> Tensor:
        output = self.__compute_linear_transformation(X, weights)
        return self.__add_bias_parameters(output, bias)

    def __compute_linear_transformation(
        self,
        X: Tensor,
        dynamic_diagonal_weights: Tensor,
    ) -> Tensor:
        if dynamic_diagonal_weights.dim() == 3:
            return torch.einsum("ij,ijk->ik", X, dynamic_diagonal_weights)
        return torch.matmul(X, dynamic_diagonal_weights)

    def __add_bias_parameters(
        self,
        X: Tensor,
        bias_params: Tensor | None = None,
    ) -> Tensor:
        if bias_params is not None:
            return X + bias_params
        return X
