import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Sequential
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.base.utils import Module
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.generators.utils.base import (
    LinearBlockStack,
    LinearBlockStackConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.utils.layers import LinearLayerConfig


def linear_stack_config(obj, output_dim: int) -> LinearBlockStackConfig:
    return LinearBlockStackConfig(
        input_dim=obj.input_dim,
        hidden_dim=obj.input_dim,
        output_dim=output_dim,
        num_layers=obj.dynamic_generators_depth,
        activation=F.relu,
        layer_norm_position=LayerNormPositionOptions.DEFAULT,
        model_type=nn.Linear,
    )


class DiagonalParametersBehaviour(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
        weight_params: Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.anti_diagonal_flag = cfg.anti_diagonal_flag
        self.dynamic_generators_depth = cfg.dynamic_generators_depth
        self.diagonal_model, self.anti_diagonal_model = self.__init_diagonal_models()
        self.padding_shape = self.__get_diagonal_padding_shape()

        if weight_params is not None:
            self.set_parameters(weight_params)

    def set_parameters(self, weight_params: Tensor) -> None:
        self.weight_params = weight_params

    def __init_diagonal_models(
        self,
    ) -> tuple[Linear | Sequential, Linear | Sequential | None]:
        if self.input_dim is None and self.output_dim is None:
            return (None, None)
        output_dim = min(self.input_dim, self.output_dim)
        cfg = linear_stack_config(self, output_dim=output_dim)
        diagonal_model = LinearBlockStack(cfg).build_model()
        anti_diagonal_model = None
        if self.anti_diagonal_flag:
            anti_diagonal_model = LinearBlockStack(cfg).build_model()
        return diagonal_model, anti_diagonal_model

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
    ) -> Tensor:
        weight_params = self.__add_diagonal_matrix(logits)
        weight_params = self.__add_anti_diagonal_matrix(logits, weight_params)
        return weight_params

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


class DynamicParametersBehaviour(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
        bias_params: Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.dynamic_bias_flag = cfg.dynamic_bias_flag
        self.dynamic_generators_depth = cfg.dynamic_generators_depth
        self.bias_model = self.__init_bias_model()
        if bias_params is not None:
            self.set_parameters(bias_params)

    def set_parameters(self, bias_params: Tensor | None = None) -> None:
        self.bias_params = bias_params

    def __init_bias_model(
        self,
    ) -> Linear | Sequential | None:
        if self.input_dim is None:
            return None
        if not self.dynamic_bias_flag:
            return None

        output_dim = 2
        cfg = linear_stack_config(self, output_dim=output_dim)
        return LinearBlockStack(cfg).build_model()

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor | None:
        if self.bias_params is None and not self.dynamic_bias_flag:
            return self.bias_params
        bias_scalars = self.bias_model(logits)
        bias_scaling_factor, bias_offset = bias_scalars.chunk(2, dim=-1)
        return bias_scaling_factor * self.bias_params + bias_offset


class DynamicBiasBehaviour(Module):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
        bias_params: Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.dynamic_bias_flag = cfg.dynamic_bias_flag
        self.dynamic_generators_depth = cfg.dynamic_generators_depth
        self.bias_model = self.__init_bias_model()
        if bias_params is not None:
            self.set_parameters(bias_params)

    def set_parameters(self, bias_params: Tensor | None = None) -> None:
        self.bias_params = bias_params

    def __init_bias_model(
        self,
    ) -> Linear | Sequential | None:
        if self.input_dim is None:
            return None
        if not self.dynamic_bias_flag:
            return None

        output_dim = 2
        cfg = linear_stack_config(self, output_dim=output_dim)
        return LinearBlockStack(cfg).build_model()

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor | None:
        if self.bias_params is None and not self.dynamic_bias_flag:
            return self.bias_params
        bias_scalars = self.bias_model(logits)
        bias_scaling_factor, bias_offset = bias_scalars.chunk(2, dim=-1)
        return bias_scaling_factor * self.bias_params + bias_offset
