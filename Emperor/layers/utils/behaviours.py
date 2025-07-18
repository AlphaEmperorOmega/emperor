import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Sequential
from Emperor.base.utils import Module
from Emperor.layers.utils.base import (
    LinearBlockStack,
    LinearBlockStackConfig,
)


class DynamicDiagonalParametersBehaviour(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        anti_diagonal_flag: bool = True,
        dynamic_bias_flag: bool = False,
        weight_params: Tensor | None = None,
        bias_params: Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.anti_diagonal_flag = anti_diagonal_flag
        self.dynamic_bias_flag = dynamic_bias_flag
        self.diagonal_model, self.anti_diagonal_model, self.bias_model = (
            self.__init_diagonal_models()
        )
        self.padding_shape = self.__get_diagonal_padding_shape()

        if weight_params is not None:
            self.set_parameters(weight_params, bias_params)

    def set_parameters(
        self, weight_params: Tensor, bias_params: Tensor | None = None
    ) -> None:
        self.weight_params = weight_params
        self.bias_params = bias_params

    def __init_diagonal_models(
        self,
    ) -> tuple[
        Linear | Sequential | None,
        Linear | Sequential | None,
        Linear | Sequential | None,
    ]:
        if self.input_dim is None and self.output_dim is None:
            return (None, None, None)
        output_dim = min(self.input_dim, self.output_dim)
        cfg = LinearBlockStackConfig(
            input_dim=self.input_dim,
            hidden_dim=self.input_dim,
            output_dim=output_dim,
            num_layers=2,
            activation=nn.ReLU,
            layer_norm_flag=False,
            linear_model=nn.Linear,
        )
        diagonal_model = LinearBlockStack(cfg).build_model()
        anti_diagonal_model = None
        if self.anti_diagonal_flag:
            anti_diagonal_model = LinearBlockStack(cfg).build_model()
        bias_model = None
        if self.dynamic_bias_flag:
            overrides = LinearBlockStackConfig(
                output_dim=2,
            )
            bias_model = LinearBlockStack(cfg, overrides).build_model()
        return diagonal_model, anti_diagonal_model, bias_model

    def __get_diagonal_padding_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def forward(
        self,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        weight_params = self.__add_diagonal_matrix(logits)
        weight_params = self.__add_anti_diagonal_matrix(logits, weight_params)
        bias_params = self.__maybe_update_bias_parameters(logits)
        return weight_params, bias_params

    def __add_diagonal_matrix(
        self,
        logits: Tensor,
    ):
        diagonal_vectors = self.diagonal_model(logits)
        diagonal_matrix = self.__convert_to_diagonal_matrix(diagonal_vectors)
        return self.weight_params + diagonal_matrix

    def __add_anti_diagonal_matrix(
        self,
        logits: Tensor,
        weight_params: Tensor,
    ) -> Tensor:
        if self.anti_diagonal_flag:
            anti_diagonal_vectors = self.anti_diagonal_model(logits)
            anti_diagonal_matrix = self.__convert_to_diagonal_matrix(
                anti_diagonal_vectors
            )
            return weight_params + anti_diagonal_matrix.flip(dims=[2])
        return weight_params

    def __convert_to_diagonal_matrix(
        self,
        vector_matrix: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(vector_matrix)
        if self.padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
        return diagonal_matrix

    def __maybe_update_bias_parameters(self, logits: Tensor) -> Tensor | None:
        if self.bias_params is not None and self.dynamic_bias_flag:
            bias_scalars = self.bias_model(logits)
            bias_scaling_factor, bias_offset = bias_scalars.chunk(2, dim=-1)
            return bias_scaling_factor * self.bias_params + bias_offset
        return self.bias_params
