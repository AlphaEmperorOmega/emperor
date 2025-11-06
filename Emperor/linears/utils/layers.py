from inspect import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.linears.utils.behaviours import (
    DiagonalParametersBehaviour,
    DiagonalParametersOptions,
    DynamicBiasBehaviour,
    DynamicBiasOptions,
    DynamicParametersBehaviour,
)
from Emperor.linears.utils.monitors import (
    DataMonitor,
    ParameterMonitor,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int = field(
        default=0,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int = field(
        default=0,
        metadata={"help": "Output dimension of the linera layer"},
    )
    bias_flag: bool = field(
        default=False,
        metadata={
            "help": "When true bias will be added to after the matrix multiplication between, the input and output"
        },
    )
    anti_diagonal_flag: bool = field(
        default=False,
        metadata={
            "help": "When `True` the `DynamicLinearLayer` will add `anti_diagonal_matrix` after linear transformation"
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
        cfg: "DynamicBiasBehaviour | LinearLayerConfig | ModelConfig",
        overrides: "DynamicBiasBehaviour | LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_model_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.bias_flag = self.cfg.bias_flag
        self.dynamic_bias_options = self.cfg.dynamic_bias_options
        self.dynamic_generators_depth = self.cfg.dynamic_generators_depth
        self.anti_diagonal_flag = self.cfg.anti_diagonal_flag
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


class DynamicLinearLayerConfig(LinearLayerConfig):
    anti_diagonal_flag: bool = field(
        default=False,
        metadata={
            "help": "When `True` the `DynamicLinearLayer` will add `anti_diagonal_matrix` after linear transformation"
        },
    )
    dynamic_generators_depth: int = field(
        default=1,
        metadata={
            "help": "When `True` a generate a `scaler` and `offset` that will be used on the `bias_parameters` for each sampele in the batch"
        },
    )
    dynamic_diagonal_options: DiagonalParametersOptions = field(
        default=DiagonalParametersOptions.DEFAULT,
        metadata={
            "help": "When `True` a generate a `scaler` and `offset` that will be used on the `bias_parameters` for each sampele in the batch"
        },
    )
    dynamic_bias_options: DynamicBiasOptions = field(
        default=DynamicBiasOptions.DEFAULT,
        metadata={
            "help": "When `True` a generate a `scaler` and `offset` that will be used on the `bias_parameters` for each sampele in the batch"
        },
    )


class DynamicLinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig | ModelConfig",
        overrides: "DynamicLinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        weight_params, bias_params = self._init_parameters()

        self.generator_model = DynamicParametersBehaviour(self.cfg, weight_params)
        self.diagonal_model = DiagonalParametersBehaviour(self.cfg, weight_params)
        self.bias_model = (
            DynamicBiasBehaviour(self.cfg, bias_params) if self.bias_flag else None
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        weight_params = self.generator_model(input_batch)
        weight_params = self.diagonal_model(input_batch)
        bias_parameters = self.bias_model(input_batch)
        output = self.__compute_linear_transformation(input_batch, weight_params)
        return self.__add_bias_parameters(output, bias_parameters)

    def __compute_linear_transformation(
        self,
        logits: Tensor,
        dynamic_diagonal_weights: Tensor,
    ) -> Tensor:
        return torch.einsum("ij,ijk->ik", logits, dynamic_diagonal_weights)

    def __add_bias_parameters(
        self,
        linear_transform: Tensor,
        bias_params: Tensor | None,
    ) -> Tensor:
        if self.bias_flag and bias_params:
            return linear_transform + bias_params
        return linear_transform


class LinearLayerWithMemoryOptions(Enum):
    CONCATENATE_VECTORS = 1
    ADD_VECTORS = 2
    WEIGHTED_SUM = 3


class MemoryLinearLayerConfig(LinearLayerConfig):
    dynamic_bias_options: DynamicBiasOptions = field(
        default=DynamicBiasOptions.DEFAULT,
        metadata={
            "help": "When `True` a generate a `scaler` and `offset` that will be used on the `bias_parameters` for each sampele in the batch"
        },
    )


class MemoryLinearLayer(LinearBase):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        config = getattr(cfg, "linear_layer_model_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.memory_dim = self.cfg.memory_dim
        # self.memory_linear_option = self.cfg.memory_linear_option
        self.bias_flag = self.cfg.bias_flag
        self.weight_params, self.bias_params = self.__init_parameter_banks(
            self.output_dim + self.memory_dim
        )
        self.memory_weight_params, self.memory_bias_params = (
            self.__init_parameter_banks(self.memory_dim)
        )
        self.memory_model = nn.Linear(self.input_dim, self.memory_dim)
        self.attentional_memory_model = nn.Linear(self.batch_size * 2, self.batch_size)

        # self.memory_weight_params = None
        # if self.memory_linear_option == LinearLayerWithMemoryOptions.WEIGHTED_SUM:
        #     self.memory_weight_params = nn.Linear(self.input_dim, 2)

    def __init_parameter_banks(self, output_dim: int | None = None):
        output_dim = output_dim or self.output_dim
        weight_shape = (self.input_dim, self.output_dim)
        weight_params = self._init_parameter_bank(weight_shape)
        bias_params = None
        if self.bias_flag:
            bias_shape = (self.output_dim,)
            bias_params = self._init_parameter_bank(bias_shape, nn.init.zeros_)
        return weight_params, bias_params

    def forward(self, input_batch: Tensor) -> Tensor:
        # TODO: This most likely does not work in the current state
        # this is implemented as an idea that can be later make to work
        # this is to give you an idea of how this should work
        # Basically:
        # - create a memory tensor that is the same shape as the input
        # - concatenate the memory with the input tensor this is goung to have, input_with_memory (batch_size * 2, input_dim)
        # - then you generate a set of weights that compresses the memory with the input tensor of shape (batch_size * 2, batch_size)
        # - you use softmax to normalize across the batch dimension
        # - you multiply this by the input_with_memory tensor, basically take a weighted sum of all tokens in the batch with the memory cells combined
        # - once you weighted the tokens across the batch_dim you shound end um with a tensor with the same shape as the original input (batch_size, input_dim)
        memory = self.memory_model(input_batch)
        input_with_memory = torch.cat((input_batch, memory), dim=0)
        attentional_weight = self.attentional_memory_model(input_with_memory.T)
        attentional_weight = F.softmax(attentional_weight, dim=-1)

        associative_memory = torch.matmul()

        return

    # def forward(self, input_batch: Tensor) -> Tensor:
    #     memory = self.memory_model(input_batch)
    #
    #     if (
    #         self.memory_linear_option
    #         == LinearLayerWithMemoryOptions.CONCATENATE_VECTORS
    #     ):
    #         inputs_with_memory = torch.cat((input_batch, memory), dim=-1)
    #         return F.linear(inputs_with_memory, self.weight_params.T, self.bias_params)
    #     elif self.memory_linear_option == LinearLayerWithMemoryOptions.ADD_VECTORS:
    #         affine_transformation = F.linear(
    #             input_batch, self.weight_params.T, self.bias_params
    #         )
    #         return affine_transformation + memory
    #     elif self.memory_linear_option == LinearLayerWithMemoryOptions.WEIGHTED_SUM:
    #         memery_weights = self.memory_weight_params(input_batch).flatten()
    #         inputs_with_memory = torch.stack(input_batch, memory)
    #         weighted_inputs = inputs_with_memory * memery_weights.unsqueeze(1)
    #         summed_inputs = torch.sum(weighted_inputs, dim=0)
    #         return summed_inputs
    #
    #     return F.linear(input_batch, self.weight_params.T, self.bias_params)
