from torch import Tensor
from typing import Callable
from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.core._validator import (
    AdaptiveParameterAugmentationValidator,
)


class AdaptiveParameterAugmentation(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._override_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.weight_config = self.cfg.weight_config
        self.diagonal_config = self.cfg.diagonal_config
        self.bias_config = self.cfg.bias_config
        self.mask_config = self.cfg.mask_config
        self.model_config = self.cfg.model_config
        AdaptiveParameterAugmentationValidator.validate(self)
        self.generator_model = self.__build_from_config(self.weight_config)
        self.diagonal_model = self.__build_from_config(self.diagonal_config)
        self.bias_model = self.__build_from_config(self.bias_config)
        self.mask_model = self.__build_from_config(self.mask_config)

    def __build_from_config(self, config: ConfigBase | None) -> Module | None:
        if config is None:
            return None
        if self.__is_model_type_disabled(config):
            return None
        if config.model_config is None:
            config.model_config = self.model_config
        overrides = type(config)(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return config.build(overrides)

    def __is_model_type_disabled(self, config: ConfigBase) -> bool:
        if not hasattr(config, "model_type"):
            return False
        if config.model_type is None:
            return False
        return config.model_type.value == 0

    def forward(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor,
    ) -> Tensor:
        weights, bias = self.__apply_adaptive_adjustments(
            weight_params, bias_params, input
        )
        weights = self.__maybe_apply_weight_mask(weights, input)
        return affine_transform_callback(weights, bias, input)

    def __apply_adaptive_adjustments(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__call_model(self.generator_model, weights, input)
        weights = self.__call_model(self.diagonal_model, weights, input)
        bias = self.__call_bias_model(self.bias_model, bias, input)
        return weights, bias

    def __maybe_apply_weight_mask(self, weights: Tensor, input: Tensor) -> Tensor:
        return self.__call_model(self.mask_model, weights, input)

    def __call_model(
        self,
        model,
        parameters: Tensor | None = None,
        input: Tensor | None = None,
    ) -> Tensor | None:
        if model is None:
            return parameters
        if parameters is None:
            return model(input)
        return model(parameters, input)

    def __call_bias_model(
        self,
        model,
        parameters: Tensor | None = None,
        input: Tensor | None = None,
    ) -> Tensor | None:
        if model is None:
            return parameters
        return model(parameters, input)
