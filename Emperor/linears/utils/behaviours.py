from enum import Enum
from inspect import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear, Sequential
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.base.utils import Module
from Emperor.generators.utils.base import (
    LinearBlockStack,
    LinearBlockStackConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.utils.layers import DynamicLinearLayerConfig, LinearLayerConfig


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


class MemoryBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        weight_params: Tensor,
    ):
        super().__init__()
        self.weight_params = weight_params
        self.memory_model = self.__init_memory_model()

    def __init_memory_model(self) -> Linear | Sequential:
        scalar_and_offset = 2
        cfg = linear_stack_config(self, output_dim=scalar_and_offset)
        return LinearBlockStack(cfg).build_model()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class OuterProductNormOptions(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    LAYER_NORM = 4


class DynamicParametersBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        weight_params: Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.weight_params = weight_params
        self.input_model = self.__init_generator_model()
        self.output_model = self.__init_generator_model()

    def __init_generator_model(self) -> Linear | Sequential:
        scalar_and_offset = 2
        cfg = linear_stack_config(self, output_dim=scalar_and_offset)
        return LinearBlockStack(cfg).build_model()

    def forward(self, inputs: Tensor) -> Tensor:
        input_vectors = self.input_model(inputs)
        output_vectors = self.output_model(inputs)
        outer_product = self.__compute_outer_product(input_vectors, output_vectors)
        return self.weight_params + outer_product

    def __compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        test_norm_option = OuterProductNormOptions.SIGMOID
        input_vectors = self.__normalize_vectors(input_vectors, test_norm_option)
        output_vectors = self.__normalize_vectors(output_vectors, test_norm_option)
        outer_product = torch.einsum("bij,bik->bijk", input_vectors, output_vectors)
        return self.__normalize_vectors(outer_product)

    def __normalize_vectors(
        self,
        outer_product: Tensor,
        norm_option: OuterProductNormOptions | None,
    ) -> Tensor:
        # TODO: Temporary nomralization just to check what's happening
        match norm_option:
            case OuterProductNormOptions.RELU:
                return F.relu(outer_product)
            case OuterProductNormOptions.TANH:
                return F.tanh(outer_product)
            case OuterProductNormOptions.SIGMOID:
                return F.sigmoid(outer_product)
            case _:
                return outer_product


class DynamicDiagonalOptions(Enum):
    DEFAULT = 0
    DIAGONAL = 1
    ANTI_DIAGONAL = 2
    DIAGONAL_AND_ANTI_DIAGONAL = 3


# TODO: Add option for a kernel to take the context
# of every token into account when computing the dynamic parameters


class DynamicDiagonalBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.model = self.__init_bias_model()

    def __init_bias_model(
        self,
    ) -> "DefaultDiagonalHandler":
        match self.dynamic_bias_option:
            case DynamicDiagonalOptions.DEFAULT:
                return DefaultDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DIAGONAL:
                return DiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.ANTI_DIAGONAL:
                return AntiDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL:
                return DiagonalAndAntiDiagonalHandler(self.cfg)
            case _:
                raise ValueError(
                    f"Unsupported `dynamic_bias_option`: {self.dynamic_bias_option}"
                )


class DefaultDiagonalHandler(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.padding_shape = self.__get_diagonal_padding_shape()

    def __get_diagonal_padding_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def _init_diagonal_model(
        self,
    ) -> Linear | Sequential:
        output_dim = min(self.input_dim, self.output_dim)
        cfg = linear_stack_config(self, output_dim=output_dim)
        return LinearBlockStack(cfg).build_model()

    def forward(self, weight_params: Tensor) -> Tensor:
        return weight_params

    def _convert_to_diagonal_matrix(
        self,
        vector_matrix: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(vector_matrix)
        if self.padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
        return diagonal_matrix


class DiagonalHandler(DefaultDiagonalHandler):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__(cfg)
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.diagonal_generator = self._init_diagonal_model()

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        diagonal_vectors = self.diagonal_generator(logits)
        diagonal_matrix = self._convert_to_diagonal_matrix(diagonal_vectors)
        return weight_params + diagonal_matrix


class AntiDiagonalHandler(DefaultDiagonalHandler):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__(cfg)
        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.diagonal_generator = self._init_diagonal_model()

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        dynamic_vectors = self.anti_diagonal_model(logits)
        diagonal_matrix = self._convert_to_diagonal_matrix(dynamic_vectors)
        anti_diagonal_matrix = diagonal_matrix.flip(dims=[2])
        return weight_params + anti_diagonal_matrix


class DiagonalAndAntiDiagonalHandler(DefaultDiagonalHandler):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__(cfg)
        self.diagonal_generator = DiagonalHandler(cfg)
        self.anti_diagonal_generator = AntiDiagonalHandler(cfg)

    def forward(self, logits: Tensor, weight_params: Tensor) -> Tensor:
        weight_params = self.diagonal_generator(logits, weight_params)
        anti_diagonal_output = self.diagonal_generator(logits, weight_params)
        return anti_diagonal_output


class DynamicBiasOptions(Enum):
    DEFAULT = 0
    SCALE_AND_OFFSET = 1
    DYNAMIC_PARAMETERS = 2


class DynamicBiasBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.dynamic_bias_option = cfg.dynamic_bias_option
        self.bias_model = self.__init_bias_model()

    def __init_bias_model(
        self,
    ) -> "DefaultBiasHandler":
        match self.dynamic_bias_option:
            case DynamicBiasOptions.DEFAULT:
                return DefaultBiasHandler(self.cfg)
            case DynamicBiasOptions.SCALE_AND_OFFSET:
                return AffineBiasTransformHandler(self.cfg)
            case DynamicBiasOptions.DYNAMIC_PARAMETERS:
                return BiasGeneratorHandler(self.cfg)
            case _:
                raise ValueError(
                    f"Unsupported `dynamic_bias_option`: {self.dynamic_bias_option}"
                )

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor | None:
        return self.bias_model(logits)


class DefaultBiasHandler(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        self.output_dim = cfg.output_dim
        self.bias_params = self.__init_blas_parameters()

    def __init_blas_parameters(self) -> Parameter:
        bias_shape = (self.output_dim,)
        return self._init_parameter_bank(bias_shape, nn.init.zeros_)

    def forward(self, logits: Tensor) -> Tensor:
        return self.bias_params


class AffineBiasTransformHandler(DefaultBiasHandler):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
    ):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.bias_flag = cfg.bias_flag
        self.bias_params = self.__init_bias_parameters()
        self.scalar_and_offset_generator = self.__init_scale_and_offset_model()

    def __init_bias_parameters(self) -> Parameter:
        bias_shape = (self.output_dim,)
        return self._init_parameter_bank(bias_shape, nn.init.zeros_)

    def __init_scale_and_offset_model(self) -> tuple[Linear | Sequential, Tensor]:
        scalar_and_offset = 2
        cfg = linear_stack_config(self, output_dim=scalar_and_offset)
        generator = LinearBlockStack(cfg).build_model()
        return generator

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor:
        bias_scalars = self.scalar_and_offset_generator(logits)
        bias_scaling_factor, bias_offset = bias_scalars.chunk(2, dim=-1)
        return bias_scaling_factor * self.bias_params + bias_offset


class BiasGeneratorHandler(DefaultBiasHandler):
    def __init__(
        self,
        cfg: "LinearLayerConfig",
    ):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.dynamic_bias_flag = cfg.dynamic_bias_flag
        self.dynamic_generators_depth = cfg.dynamic_generators_depth
        self.bias_generator = self.__init_bias_model()

    def __init_bias_model(
        self,
    ) -> Linear | Sequential | None:
        cfg = linear_stack_config(self, output_dim=self.output_dim)
        return LinearBlockStack(cfg).build_model()

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor | None:
        return self.bias_generator(logits)


# class DiagonalParametersBehaviour(Module):
#     def __init__(
#         self,
#         cfg: "DynamicLinearLayerConfig",
#     ):
#         super().__init__()
#         self.input_dim = cfg.input_dim
#         self.output_dim = cfg.output_dim
#         self.anti_diagonal_flag = cfg.anti_diagonal_flag
#         self.dynamic_generators_depth = cfg.dynamic_generators_depth
#         self.diagonal_model, self.anti_diagonal_model = self.__init_diagonal_models()
#         self.padding_shape = self.__get_diagonal_padding_shape()
#
#         if weight_params is not None:
#             self.set_parameters(weight_params)
#
#     def set_parameters(self, weight_params: Tensor) -> None:
#         self.weight_params = weight_params
#
#     def __init_diagonal_models(
#         self,
#     ) -> tuple[Linear | Sequential, Linear | Sequential | None]:
#         if self.input_dim is None and self.output_dim is None:
#             return (None, None)
#         output_dim = min(self.input_dim, self.output_dim)
#         cfg = linear_stack_config(self, output_dim=output_dim)
#
#         diagonal_model = LinearBlockStack(cfg).build_model()
#         anti_diagonal_model = None
#         if self.anti_diagonal_flag:
#             anti_diagonal_model = LinearBlockStack(cfg).build_model()
#         return diagonal_model, anti_diagonal_model
#
#     def __get_diagonal_padding_shape(self) -> tuple | None:
#         diagonal_padding_shape = None
#         if self.input_dim != self.output_dim:
#             padding_size = abs(self.input_dim - self.output_dim)
#             diagonal_padding_shape = (0, padding_size, 0, 0)
#             if self.input_dim > self.output_dim:
#                 diagonal_padding_shape = (0, 0, 0, padding_size)
#         return diagonal_padding_shape
#
#     def forward(
#         self,
#         logits: Tensor,
#     ) -> Tensor:
#         weight_params = self.__add_diagonal_matrix(logits)
#         weight_params = self.__add_anti_diagonal_matrix(logits, weight_params)
#         return weight_params
#
#     def __add_diagonal_matrix(
#         self,
#         logits: Tensor,
#     ):
#         diagonal_vectors = self.diagonal_model(logits)
#         diagonal_matrix = self.__convert_to_diagonal_matrix(diagonal_vectors)
#         return self.weight_params + diagonal_matrix
#
#     def __add_anti_diagonal_matrix(
#         self,
#         logits: Tensor,
#         weight_params: Tensor,
#     ) -> Tensor:
#         if self.anti_diagonal_flag:
#             anti_diagonal_vectors = self.anti_diagonal_model(logits)
#             anti_diagonal_matrix = self.__convert_to_diagonal_matrix(
#                 anti_diagonal_vectors
#             )
#             return weight_params + anti_diagonal_matrix.flip(dims=[2])
#         return weight_params
#
#     def __convert_to_diagonal_matrix(
#         self,
#         vector_matrix: Tensor,
#     ) -> Tensor:
#         diagonal_matrix = torch.diag_embed(vector_matrix)
#         if self.padding_shape is not None:
#             diagonal_matrix = F.pad(diagonal_matrix, self.padding_shape)
#         return diagonal_matrix
