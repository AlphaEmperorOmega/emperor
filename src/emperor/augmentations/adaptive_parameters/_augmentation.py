from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.augmentations.adaptive_parameters._config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._grouping import (
    AdaptiveGroupPlan,
    build_adaptive_group_plan,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveParameterAugmentationValidator,
)
from emperor.config import ConfigBase
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers import RowLayout


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
        self.grouping_scope = self.cfg.grouping_scope
        self.group_count = self.cfg.group_count
        self.VALIDATOR.validate(self)
        self.weight_model = self.__build_from_config(self.weight_config)
        self.diagonal_model = self.__build_from_config(self.diagonal_config)
        self.bias_model = self.__build_from_config(self.bias_config)
        self.mask_model = self.__build_from_config(self.mask_config)

    @property
    def adaptive_parameter_grouping_enabled(self) -> bool:
        return self.grouping_scope != AdaptiveParameterGroupingScopeOptions.DISABLED

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
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor,
        *,
        row_layout: "RowLayout | None" = None,
    ) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(
            self, affine_transform_callback, weight_params, bias_params, input
        )
        if self.grouping_scope != AdaptiveParameterGroupingScopeOptions.DISABLED:
            return self.__apply_grouped_augmentation(
                affine_transform_callback,
                weight_params,
                bias_params,
                input,
                row_layout=row_layout,
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
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor,
        *,
        row_layout: "RowLayout | None",
    ) -> Tensor:
        self.VALIDATOR.validate_grouped_forward_inputs(
            weight_params, bias_params, row_layout
        )
        group_plan = build_adaptive_group_plan(
            input, self.grouping_scope, self.group_count, row_layout
        )
        grouped_output = self.__apply_augmentation(
            affine_transform_callback,
            weight_params,
            bias_params,
            group_plan,
        )
        return group_plan.restore(grouped_output)

    def __apply_augmentation(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor | AdaptiveGroupPlan,
    ) -> Tensor:
        parameter_generation_context = self.__prepare_input(input)
        weights, bias = self.__prepare_parameters(
            weight_params, bias_params, parameter_generation_context
        )
        if isinstance(input, AdaptiveGroupPlan):
            input = input.grouped_members
        return affine_transform_callback(weights, bias, input)

    def __prepare_input(self, input: Tensor | AdaptiveGroupPlan) -> Tensor:
        valid_members = None
        if isinstance(input, AdaptiveGroupPlan):
            valid_members = input.valid_members
            input = input.grouped_members

        if valid_members is not None:
            invalid_member_mask = ~valid_members.unsqueeze(-1)
            input = input.masked_fill(invalid_member_mask, 0)
        if input.dim() == 2:
            return input
        return input.sum(dim=1)

    def __prepare_parameters(
        self,
        weight_params: Tensor,
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
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__call_model(self.weight_model, weights, input)
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
