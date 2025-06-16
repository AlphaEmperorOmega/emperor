import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from Emperor.base.utils import Module


class DynamicDiagonalParametersBehaviour(Module):
    def __init__(
        self,
        weight_params: Tensor,
        bias_params: Tensor | None = None,
        anti_diagonal_flag: bool = True,
    ):
        super().__init__()

        self.input_dim, self.output_dim = weight_params.shape
        self.weight_params = weight_params
        self.bias_params = bias_params
        self.anti_diagonal_flag = anti_diagonal_flag
        self.diagonal_model, self.anti_diagonal_model = self.__init_diagonal_models()
        self.padding_shape = self.__get_diagonal_padding_shape()

    def __get_diagonal_padding_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def __init_diagonal_models(self) -> tuple[nn.Linear, nn.Linear | None]:
        output_dim = min(self.input_dim, self.output_dim)
        diagonal_shape = (self.input_dim, output_dim)
        diagonal_model = nn.Linear(*diagonal_shape)
        anti_diagonal_model = None
        if self.anti_diagonal_flag:
            anti_diagonal_shape = (self.input_dim, output_dim)
            anti_diagonal_model = nn.Linear(*anti_diagonal_shape)
        return diagonal_model, anti_diagonal_model

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor:
        added_diagonal_matrix = self.__add_diagonal_matrix(logits)
        added_anti_diagonal_matrix = self.__add_anti_diagonal_matrix(
            logits, added_diagonal_matrix
        )
        return self.__compute_affine_transformation(logits, added_anti_diagonal_matrix)

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

    def __compute_affine_transformation(
        self,
        logits: Tensor,
        dynamic_weight_params: Tensor,
    ) -> Tensor:
        linear_transform = torch.einsum("ij,ijk->ik", logits, dynamic_weight_params)
        if self.bias_params is not None:
            return linear_transform + self.bias_params
        return linear_transform
