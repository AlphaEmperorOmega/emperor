import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module
from torch import Tensor
from Emperor.components.parameter_generators.utils.behaviours import (
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
            self.bias_params + 2,
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
