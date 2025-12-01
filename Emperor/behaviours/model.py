from Emperor.behaviours.utils.behaviours import (
    DynamicBiasSelector,
    DynamicDiagonalSelector,
    DynamicMemorySelector,
    DynamicParametersBehaviour,
)
from Emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
)
from Emperor.base.utils import Module
from Emperor.linears.utils.layers import AdaptiveLinearLayerConfig


class AdaptiveParameterModel(Module):
    def __init__(
        self,
        cfg: "AdaptiveLinearLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.bias_flag = self.cfg.bias_flag
        self.generator_depth = self.cfg.generator_depth
        self.diagonal_option = self.cfg.diagonal_option
        self.memory_option = self.cfg.memory_option
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option
        self.bias_option = self.cfg.bias_option
        self.generator_model = self.__init_generator_model()
        self.diagonal_model = self.__init_diagonal_model()
        self.memory_model = self.__init_memory_model()
        self.bias_model = self.__init_bias_model()

    def __init_generator_model(self) -> DynamicParametersBehaviour | None:
        is_valid_flag = self.generator_depth != DynamicDepthOptions.DISABLED
        return self.__init_model(is_valid_flag, DynamicParametersBehaviour)

    def __init_diagonal_model(self) -> DynamicDiagonalSelector | None:
        is_valid_flag = self.diagonal_option != DynamicDiagonalOptions.DISABLED
        return self.__init_model(is_valid_flag, DynamicDiagonalSelector)

    def __init_memory_model(self) -> DynamicMemorySelector | None:
        is_valid_flag = self.memory_option != LinearMemoryOptions.DISABLED
        return self.__init_model(is_valid_flag, DynamicMemorySelector)

    def __init_bias_model(self) -> DynamicBiasSelector | None:
        is_disabled = self.bias_option != DynamicBiasOptions.DISABLED
        is_valid_flag = is_disabled and self.bias_flag
        return self.__init_model(is_valid_flag, DynamicBiasSelector)

    def __init_model(self, is_valid_flag: bool, model_class: object) -> object | None:
        if is_valid_flag:
            overrides = AdaptiveLinearLayerConfig(
                input_dim=self.input_dim, output_dim=self.output_dim
            )
            return model_class(self.cfg, overrides)
        return None

    def compute_dynamic_parameters(
        self,
        affine_transform_callback: Callable,
        weight_params: Tensor,
        bias_params: Tensor | None,
        input: Tensor,
    ) -> Tensor:
        input = self.__apply_memory(input, LinearMemoryPositionOptions.BEFORE_AFFINE)
        weights, bias = self.__update_parameters(weight_params, bias_params, input)
        output = affine_transform_callback(weights, bias, input)
        output = self.__apply_memory(output, LinearMemoryPositionOptions.AFTER_AFFINE)
        return output

    def __apply_memory(
        self,
        input: Tensor,
        position: LinearMemoryPositionOptions,
    ) -> Tensor:
        if self.memory_model and self.memory_position_option == position:
            return self.__call_model(self.memory_model, None, input)
        return input

    def __update_parameters(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__call_model(self.generator_model, weights, input)
        weights = self.__call_model(self.diagonal_model, weights, input)
        bias = self.__call_model(self.bias_model, bias, input)
        return weights, bias

    def __call_model(
        self, model, parameters: Tensor | None, input: Tensor
    ) -> Tensor | None:
        if model is None:
            return parameters
        if parameters is None:
            return model(input)
        return model(parameters, input)
