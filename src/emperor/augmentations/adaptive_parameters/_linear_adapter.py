from __future__ import annotations

import torch
from torch import Tensor

from emperor.augmentations.adaptive_parameters._config import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveLinearValidator,
)
from emperor.linears import LinearAbstract


class AdaptiveLinearLayer(LinearAbstract):
    VALIDATOR = AdaptiveLinearValidator

    def __init__(
        self,
        cfg: AdaptiveLinearLayerConfig,
        overrides: AdaptiveLinearLayerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.adaptive_augmentation_config = self.cfg.adaptive_augmentation_config
        self.has_adaptive_augmentation: bool = self.__has_adaptive_augmentation()
        self.adaptive_behaviour = self.__init_behaviour()

    def __has_adaptive_augmentation(self) -> bool:
        cfg = self.adaptive_augmentation_config
        return any(
            config is not None
            for config in (
                cfg.diagonal_config,
                cfg.weight_config,
                cfg.bias_config,
                cfg.mask_config,
            )
        )

    def __init_behaviour(self):
        if not self.has_adaptive_augmentation:
            return None

        overrides = AdaptiveParameterAugmentationConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.adaptive_augmentation_config.build(overrides)

    def forward(self, X: Tensor) -> Tensor:
        self.VALIDATOR.validate_input_is_2d(X)
        if not self.has_adaptive_augmentation:
            return self._compute_affine_transformation_callback(
                self.weight_params, self.bias_params, X
            )
        return self.adaptive_behaviour(
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

    def __compute_linear_transformation(self, X: Tensor, weights: Tensor) -> Tensor:
        if weights.dim() == 3:
            return torch.einsum("ij,ijk->ik", X, weights)
        return torch.matmul(X, weights)

    def __add_bias_parameters(
        self, X: Tensor, bias_params: Tensor | None = None
    ) -> Tensor:
        if bias_params is not None:
            return X + bias_params
        return X
