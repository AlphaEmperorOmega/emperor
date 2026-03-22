from torch import Tensor
from dataclasses import dataclass, field
from typing import Callable, TypeVar
from emperor.base.utils import ConfigBase, Module
from emperor.behaviours.utils.behaviours import (
    DynamicBiasFactory,
    DynamicDiagonalFactory,
    DynamicMemoryFactory,
    DynamicParametersBehaviour,
)
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from emperor.behaviours.utils._validator import AdaptiveParameterBehaviourValidator

ModuleType = TypeVar("ModuleType", bound=Module)


@dataclass
class AdaptiveParameterBehaviourConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linear layer"},
    )
    generator_depth: DynamicDepthOptions | None = field(
        default=None,
        metadata={
            "help": "Depth of the generator network that produces input-dependent weight adjustments."
        },
    )
    diagonal_option: DynamicDiagonalOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the weight matrix diagonal."},
    )
    bias_option: DynamicBiasOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the bias vector."},
    )
    memory_option: LinearMemoryOptions | None = field(
        default=None,
        metadata={
            "help": "Blends a learned memory representation with the linear layer input or output."
        },
    )
    memory_size_option: LinearMemorySizeOptions | None = field(
        default=None,
        metadata={"help": "Size of the learned memory representation."},
    )
    memory_position_option: LinearMemoryPositionOptions | None = field(
        default=None,
        metadata={"help": "Controls when memory is applied in the computation."},
    )


class AdaptiveParameterBehaviour(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.generator_depth = self.cfg.generator_depth
        self.diagonal_option = self.cfg.diagonal_option
        self.memory_option = self.cfg.memory_option
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option
        self.bias_option = self.cfg.bias_option
        self.validator = AdaptiveParameterBehaviourValidator(self)
        self.generator_model = self.__init_generator_model()
        self.diagonal_model = self.__init_diagonal_model()
        self.memory_model = self.__init_memory_model()
        self.bias_model = self.__init_bias_model()

    def __init_generator_model(self) -> Module | None:
        is_valid_flag = self.generator_depth != DynamicDepthOptions.DISABLED
        return self.__init_model(is_valid_flag, DynamicParametersBehaviour)

    def __init_diagonal_model(self) -> Module | None:
        is_valid_flag = self.diagonal_option != DynamicDiagonalOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicDiagonalFactory)

    def __init_memory_model(self) -> Module | None:
        is_valid_flag = self.memory_option != LinearMemoryOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicMemoryFactory)

    def __init_bias_model(self) -> Module | None:
        is_valid_flag = self.bias_option != DynamicBiasOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicBiasFactory)

    def __init_model(
        self, is_valid_flag: bool, model_class: type[ModuleType]
    ) -> ModuleType | None:
        from emperor.linears.utils.config import LinearLayerConfig

        if not is_valid_flag:
            return None

        overrides = LinearLayerConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        return model_class(self.cfg, overrides)

    def __build_model(
        self, is_valid_flag: bool, factory_class: type[Module]
    ) -> Module | None:
        from emperor.linears.utils.config import LinearLayerConfig

        if not is_valid_flag:
            return None

        overrides = LinearLayerConfig(
            input_dim=self.input_dim, output_dim=self.output_dim
        )
        return factory_class(self.cfg, overrides).build()

    def compute_adaptive_parameters(
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
