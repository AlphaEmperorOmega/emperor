from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.augmentations.adaptive_parameters._config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveParameterAugmentationValidator,
)
from emperor.config import ConfigBase
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._grouping.base import (
        GrouperAbstract,
    )


class AdaptiveParameterAugmentation(Module):
    VALIDATOR = AdaptiveParameterAugmentationValidator

    def __init__(
        self,
        cfg: AdaptiveParameterAugmentationConfig,
        overrides: AdaptiveParameterAugmentationConfig | None = None,
    ):
        super().__init__()
        self.cfg: AdaptiveParameterAugmentationConfig = self._override_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.weight_config = self.cfg.weight_config
        self.diagonal_config = self.cfg.diagonal_config
        self.bias_config = self.cfg.bias_config
        self.mask_config = self.cfg.mask_config
        self.model_config = self.cfg.model_config
        self.grouping_config = self.cfg.grouping_config
        self.VALIDATOR.validate(self)
        self.grouper = self.__build_grouper()
        self.weight_model = self.__build_from_config(self.weight_config)
        self.diagonal_model = self.__build_from_config(self.diagonal_config)
        self.bias_model = self.__build_from_config(self.bias_config)
        self.mask_model = self.__build_from_config(self.mask_config)

    @property
    def adaptive_parameter_grouping_enabled(self) -> bool:
        return self.grouping_config is not None

    def __build_grouper(self) -> "GrouperAbstract | None":
        if self.grouping_config is None:
            return None
        overrides = type(self.grouping_config)(feature_dim=self.input_dim)
        return self.grouping_config.build(overrides)

    def __build_from_config(self, config: ConfigBase | None) -> Module | None:
        if config is None:
            return None
        config = deepcopy(config)
        if config.model_config is None:
            config.model_config = self.model_config
        overrides = type(config)(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return config.build(overrides)

    def forward(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor | None,
        bias_params: Tensor | None,
        input: Tensor,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(
            self, affine_transform_callback, weight_params, bias_params, input
        )
        if self.grouping_config is not None:
            return self.__apply_grouped_augmentation(
                affine_transform_callback,
                weight_params,
                bias_params,
                input,
            )
        return self.__apply_augmentation(
            affine_transform_callback,
            weight_params,
            bias_params,
            input,
        )

    def __apply_grouped_augmentation(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor | None,
        bias_params: Tensor | None,
        input: Tensor,
    ) -> Tensor:
        self.VALIDATOR.validate_grouped_forward_inputs(weight_params, bias_params)
        if self.grouping_config.chunk_size is not None and input.size(0) == 0:
            return input[:, :1].expand(0, self.output_dim)
        context, group_plan = self.grouper(input)
        grouped_output = self.__apply_augmentation(
            affine_transform_callback,
            weight_params,
            bias_params,
            group_plan.grouped_members,
            context,
        )
        return group_plan.restore(grouped_output, output_dim=self.output_dim)

    def __apply_augmentation(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor | None,
        bias_params: Tensor | None,
        input: Tensor,
        parameter_generation_context: Tensor | None = None,
    ) -> Tensor:
        if parameter_generation_context is None:
            parameter_generation_context = input
        weights, bias = self.__prepare_parameters(
            weight_params, bias_params, parameter_generation_context
        )
        return affine_transform_callback(weights, bias, input)

    def __prepare_parameters(
        self,
        weight_params: Tensor | None,
        bias_params: Tensor | None,
        parameter_generation_context: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        weights, bias = self.__apply_adaptive_adjustments(
            weight_params, bias_params, parameter_generation_context
        )
        weights = self.__maybe_apply_weight_mask(weights, parameter_generation_context)
        self.VALIDATOR.validate_generated_parameters(
            self, weights, bias, parameter_generation_context
        )
        return weights, bias

    def __apply_adaptive_adjustments(
        self, weights: Tensor | None, bias: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        if self.weight_model is not None:
            weights = self.weight_model(weights, input)
        weights = self.__call_model(self.diagonal_model, weights, input)
        bias = self.__call_bias_model(self.bias_model, bias, input)
        return weights, bias

    def __maybe_apply_weight_mask(self, weights: Tensor, input: Tensor) -> Tensor:
        return self.__call_model(self.mask_model, weights, input)

    def __call_model(
        self,
        model: Module | None,
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
