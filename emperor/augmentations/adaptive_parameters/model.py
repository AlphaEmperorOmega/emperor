from torch import Tensor
from typing import Callable
from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.core.handlers.weight import (
    WeightHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.handlers.bias import (
    BiasHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.handlers.diagonal import (
    DiagonalHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.handlers.memory import (
    MemoryHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.handlers.mask import (
    MaskHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    RowMaskOptions,
)
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
        self.weight_option = self.cfg.weight_option
        self.weight_normalization = self.cfg.weight_normalization
        self.generator_depth = self.cfg.generator_depth
        self.diagonal_option = self.cfg.diagonal_option
        self.memory_option = self.cfg.memory_option
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option
        self.bias_flag = self.cfg.bias_flag
        self.bias_option = self.cfg.bias_option
        self.row_mask_option = self.cfg.row_mask_option
        self.mask_dimension_option = self.cfg.mask_dimension_option
        self.validator = AdaptiveParameterAugmentationValidator(self)
        self.generator_model = self.__init_generator_model()
        self.diagonal_model = self.__init_diagonal_model()
        self.memory_model = self.__init_memory_model()
        self.bias_model = self.__init_bias_model()
        self.row_mask_model = self.__init_row_mask_model()

    def __init_generator_model(self) -> Module | None:
        if self.weight_config is not None:
            return None
        overrides = WeightHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.weight_cfg.build(overrides)

    def __init_diagonal_model(self) -> Module | None:
        if self.diagonal_option == DynamicDiagonalOptions.DISABLED:
            return None
        diagonal_cfg = self.cfg.diagonal_config or DiagonalHandlerConfig()
        overrides = DiagonalHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return diagonal_cfg.build(overrides)

    def __init_memory_model(self) -> Module | None:
        if self.memory_option == LinearMemoryOptions.DISABLED:
            return None
        memory_cfg = self.cfg.memory_config or MemoryHandlerConfig()
        overrides = MemoryHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return memory_cfg.build(overrides)

    def __init_bias_model(self) -> Module | None:
        is_valid_not_disabled_flag = self.bias_option != DynamicBiasOptions.DISABLED
        is_valid_flag = self.bias_flag and is_valid_not_disabled_flag
        if not is_valid_flag:
            return None
        bias_cfg = self.cfg.bias_config or BiasHandlerConfig()
        overrides = BiasHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return bias_cfg.build(overrides)

    def __init_row_mask_model(self) -> Module | None:
        if self.row_mask_option == RowMaskOptions.DISABLED:
            return None
        mask_cfg = self.cfg.mask_config or MaskHandlerConfig()
        overrides = MaskHandlerConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return mask_cfg.build(overrides)

    def compute_adaptive_parameters(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor,
    ) -> Tensor:
        input = self.__maybe_apply_memory(
            input, LinearMemoryPositionOptions.BEFORE_AFFINE
        )
        weights, bias = self.__apply_adaptive_adjustments(
            weight_params, bias_params, input
        )
        weights = self.__maybe_apply_weight_mask(weights, input)
        output = affine_transform_callback(weights, bias, input)
        output = self.__maybe_apply_memory(
            output, LinearMemoryPositionOptions.AFTER_AFFINE
        )
        return output

    def __maybe_apply_weight_mask(self, weights: Tensor, input: Tensor) -> Tensor:
        return self.__call_model(self.row_mask_model, weights, input)

    def __maybe_apply_memory(
        self,
        input: Tensor,
        position: LinearMemoryPositionOptions,
    ) -> Tensor:
        if self.memory_model and self.memory_position_option == position:
            return self.__call_model(self.memory_model, None, input)
        return input

    def __apply_adaptive_adjustments(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__call_model(self.generator_model, weights, input)
        weights = self.__call_model(self.diagonal_model, weights, input)
        bias = self.__call_bias_model(self.bias_model, bias, input)
        return weights, bias

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
