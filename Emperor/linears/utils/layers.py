import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from inspect import Parameter
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.linears.utils.enums import (
    DynamicDepthOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
    DynamicBiasOptions,
    DynamicDiagonalOptions,
)
from Emperor.linears.utils.behaviours import (
    DynamicBiasSelector,
    DynamicDiagonalSelector,
    DynamicMemorySelector,
    DynamicParametersBehaviour,
)
from Emperor.linears.utils.monitors import (
    DataMonitor,
    ParameterMonitor,
)

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linera layer"},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When true bias will be added to after the matrix multiplication between, the input and output"
        },
    )
    data_monitor: type[DataMonitor] | None = field(
        default=None,
        metadata={"help": ""},
    )
    parameter_monitor: type[ParameterMonitor] | None = field(
        default=None,
        metadata={"help": ""},
    )


class LinearBase(Module):
    def __init__(
        self,
        cfg: "AdaptiveLinearLayerConfig | LinearLayerConfig | ModelConfig",
        overrides: "AdaptiveLinearLayerConfig | LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.bias_flag = self.cfg.bias_flag
        self.data_monitor: "DataMonitor" = self.construct(self.cfg.data_monitor)
        self.parameter_monitor: "ParameterMonitor" = self.construct(
            self.cfg.parameter_monitor
        )

    def _init_parameters(self):
        weight_params = self.__init_weight_parameters()
        bias_params = self.__init_bias_parameters()
        return weight_params, bias_params

    def __init_weight_parameters(self) -> Parameter:
        weight_shape = (self.input_dim, self.output_dim)
        return self._init_parameter_bank(weight_shape)

    def __init_bias_parameters(self) -> Parameter | None:
        if not self.bias_flag:
            return None
        bias_shape = (self.output_dim,)
        return self._init_parameter_bank(bias_shape, nn.init.zeros_)

    def _update_data_monitor(self, input_batch: Tensor, output_batch: Tensor) -> None:
        self.data_monitor and self.data_monitor.update(input_batch, output_batch)

    def _update_parameter_monitor(self) -> None:
        self.data_monitor and self.data_monitor.update(
            self.weight_params, self.bias_params
        )


class LinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.weight_params, self.bias_params = self._init_parameters()

    def forward(self, input_batch: Tensor) -> Tensor:
        output = F.linear(input_batch, self.weight_params.T, self.bias_params)
        self._update_data_monitor(input_batch, output)
        self._update_parameter_monitor()
        return output


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    generator_depth: DynamicDepthOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )
    diagonal_option: DynamicDiagonalOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )
    bias_option: DynamicBiasOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )
    memory_option: LinearMemoryOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )
    memory_size_option: LinearMemorySizeOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )
    memory_position_option: LinearMemoryPositionOptions | None = field(
        default=None,
        metadata={
            "help": "",
        },
    )


class AdaptiveLinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "AdaptiveLinearLayerConfig | ModelConfig",
        overrides: "AdaptiveLinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.weight_params, self.bias_params = self._init_parameters()
        self.parameter_manager = DynamicParameterManager(self.cfg)

    def forward(self, input: Tensor) -> Tensor:
        output = self.parameter_manager.compute_dynamic_parameters(
            self.compute_dynamic_afine_transformation,
            self.weight_params,
            self.bias_params,
            input,
        )
        return output

    def compute_dynamic_afine_transformation(
        self, weights: Tensor, bias: Tensor | None, input: Tensor
    ) -> Tensor:
        output = self.__compute_linear_transformation(input, weights)
        return self.__add_bias_parameters(output, bias)

    def __compute_linear_transformation(
        self,
        logits: Tensor,
        dynamic_diagonal_weights: Tensor,
    ) -> Tensor:
        if dynamic_diagonal_weights.dim() == 3:
            return torch.einsum("ij,ijk->ik", logits, dynamic_diagonal_weights)
        return torch.matmul(logits, dynamic_diagonal_weights)

    def __add_bias_parameters(
        self,
        linear_transform: Tensor,
        bias_params: Tensor | None = None,
    ) -> Tensor:
        if self.bias_flag and bias_params is not None:
            return linear_transform + bias_params
        return linear_transform


class DynamicParameterManager(Module):
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
