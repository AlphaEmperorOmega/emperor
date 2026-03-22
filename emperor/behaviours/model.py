from torch import Tensor
from dataclasses import dataclass, field
from typing import Callable
from emperor.base.utils import ConfigBase, Module
from emperor.behaviours.utils.factory import (
    DynamicBiasFactory,
    DynamicDiagonalFactory,
    DynamicMemoryFactory,
    DynamicWeightFactory,
    RowMaskFactory,
)
from emperor.behaviours.options import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
    RowMaskOptions,
    WeightNormalizationOptions,
)
from emperor.behaviours.utils._validator import AdaptiveParameterBehaviourValidator

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
    weight_option: DynamicWeightOptions | None = field(
        default=None,
        metadata={
            "help": "Selects the weight handler type for input-dependent weight adjustments."
        },
    )
    weight_normalization: WeightNormalizationOptions | None = field(
        default=None,
        metadata={
            "help": "Normalization applied to vectors before the outer product computation."
        },
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
    row_mask_option: RowMaskOptions | None = field(
        default=None,
        metadata={
            "help": "Input-dependent row masking of the weight matrix after weight updates."
        },
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
        self.weight_option = self.cfg.weight_option
        self.weight_normalization = self.cfg.weight_normalization
        self.generator_depth = self.cfg.generator_depth
        self.diagonal_option = self.cfg.diagonal_option
        self.memory_option = self.cfg.memory_option
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option
        self.bias_option = self.cfg.bias_option
        self.row_mask_option = self.cfg.row_mask_option
        self.validator = AdaptiveParameterBehaviourValidator(self)
        self.generator_model = self.__init_generator_model()
        self.diagonal_model = self.__init_diagonal_model()
        self.memory_model = self.__init_memory_model()
        self.bias_model = self.__init_bias_model()
        self.row_mask_model = self.__init_row_mask_model()

    def __init_generator_model(self) -> Module | None:
        is_valid_flag = self.generator_depth != DynamicDepthOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicWeightFactory)

    def __init_diagonal_model(self) -> Module | None:
        is_valid_flag = self.diagonal_option != DynamicDiagonalOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicDiagonalFactory)

    def __init_memory_model(self) -> Module | None:
        is_valid_flag = self.memory_option != LinearMemoryOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicMemoryFactory)

    def __init_bias_model(self) -> Module | None:
        is_valid_flag = self.bias_option != DynamicBiasOptions.DISABLED
        return self.__build_model(is_valid_flag, DynamicBiasFactory)

    def __init_row_mask_model(self) -> Module | None:
        is_valid_flag = self.row_mask_option != RowMaskOptions.DISABLED
        return self.__build_model(is_valid_flag, RowMaskFactory)

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
        weights = self.__call_model(self.row_mask_model, weights, input)
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
