import torch
import torch.nn as nn
from enum import Enum
import torch.nn.functional as F
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module
from torch import Tensor
from Emperor.layers.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class LinearLayerConfig(DataClassBase):
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
    anti_diagonal_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the `DynamicDiagonalLinearLayer` will add `anti_diagonal_matrix` after linear transformation"
        },
    )
    dynamic_bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` a generate a `scaler` and `offset` that will be used on the `bias_parameters` for each sampele in the batch"
        },
    )


class LinearLayer(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_model_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.bias_flag = self.cfg.bias_flag
        self.dynamic_bias_flag = self.cfg.dynamic_bias_flag
        self.weight_params, self.bias_params = self.__init_parameter_banks()

    def __init_parameter_banks(self):
        weight_shape = (self.input_dim, self.output_dim)
        weight_params = self._init_parameter_bank(weight_shape)
        bias_params = None
        if self.bias_flag:
            bias_shape = (self.output_dim,)
            bias_params = self._init_parameter_bank(bias_shape, nn.init.zeros_)
        return weight_params, bias_params

    def forward(self, input_batch: Tensor) -> Tensor:
        return F.linear(input_batch, self.weight_params.T, self.bias_params)


class DynamicDiagonalLinearLayer(LinearLayer):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.anti_diagonal_flag = self.cfg.anti_diagonal_flag
        self.dynamic_diagonal_params_model = DynamicDiagonalParametersBehaviour(
            self.input_dim,
            self.output_dim,
            self.anti_diagonal_flag,
            self.dynamic_bias_flag,
            self.weight_params,
            self.bias_params,
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        diagonal_matrix, bias_parameters = self.dynamic_diagonal_params_model(
            input_batch
        )
        output = self.__compute_linear_transformation(input_batch, diagonal_matrix)
        return self.__add_bias_parameters(output, bias_parameters)

    def __compute_linear_transformation(
        self,
        logits: Tensor,
        dynamic_weight_params: Tensor,
    ) -> Tensor:
        return torch.einsum("ij,ijk->ik", logits, dynamic_weight_params)

    def __add_bias_parameters(
        self,
        linear_transform: Tensor,
        bias_params: Tensor | None,
    ) -> Tensor:
        if self.bias_flag:
            return linear_transform + bias_params
        return linear_transform


class LinearLayerWithMemoryOptions(Enum):
    CONCATENATE_VECTORS = 1
    ADD_VECTORS = 2
    WEIGHTED_SUM = 3


class LinearLayerWithMemory(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig | ModelConfig",
        overrides: "LinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_model_config", cfg)
        self.cfg: "LinearLayerConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.memory_dim = self.cfg.memory_dim
        # self.memory_linear_option = self.cfg.memory_linear_option
        self.bias_flag = self.cfg.bias_flag
        self.dynamic_bias_flag = self.cfg.dynamic_bias_flag
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
