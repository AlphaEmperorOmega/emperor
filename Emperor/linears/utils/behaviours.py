import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from torch import Tensor
from torch.nn import Linear, Sequential
from Emperor.base.utils import Module
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.linears.utils.enums import DynamicBiasOptions, DynamicDiagonalOptions
from Emperor.linears.utils.handlers.parameter import DepthMappingLayerStack
from Emperor.linears.utils.handlers.bias import (
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    DefaultBiasHandler,
    AffineBiasTransformHandler,
    ElementwiseBiasHandler,
)
from Emperor.linears.utils.handlers.diagonal import (
    AntiDiagonalHandler,
    DefaultDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
    DiagonalHandler,
    DiagonalHandlerAbstract,
)
from Emperor.base.layer import (
    LayerStackConfig,
    LinearLayerStack,
)

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.linears.utils.handlers.bias import DefaultBiasHandler
    from Emperor.linears.utils.layers import DynamicLinearLayerConfig


def linear_stack_config(obj, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=obj.input_dim,
        hidden_dim=obj.input_dim,
        output_dim=output_dim,
        num_layers=obj.dynamic_generators_depth,
        activation=F.relu,
        layer_norm_position=LayerNormPositionOptions.DEFAULT,
        model_type=LinearLayer,
    )


class MemoryBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        weight_params: Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.weight_params = weight_params
        self.memory_model = self.__init_memory_model()

    def __init_memory_model(self) -> Linear | Sequential:
        scalar_and_offset = 2
        cfg = linear_stack_config(self, output_dim=scalar_and_offset)
        return LinearLayerStack(cfg).build_model()

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

    def __init_generator_model(self) -> DepthMappingLayerStack:
        scalar_and_offset = 2
        cfg = linear_stack_config(self, output_dim=scalar_and_offset)
        return DepthMappingLayerStack(cfg)

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


# TODO: Add option for a kernel to take the context
# of every token into account when computing the dynamic parameters
class DynamicDiagonalSelector(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = config
        self.main_config = cfg
        self.diagonal_option = self.cfg.diagonal_option
        self.model = self.__init_bias_model()

    def __init_bias_model(
        self,
    ) -> DiagonalHandlerAbstract:
        match self.diagonal_option:
            case DynamicDiagonalOptions.DEFAULT:
                return DefaultDiagonalHandler(self.main_config)
            case DynamicDiagonalOptions.DIAGONAL:
                return DiagonalHandler(self.main_config)
            case DynamicDiagonalOptions.ANTI_DIAGONAL:
                return AntiDiagonalHandler(self.main_config)
            case DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL:
                return DiagonalAndAntiDiagonalHandler(self.main_config)
            case _:
                raise ValueError(
                    f"Unsupported `dynamic_bias_option`: {self.dynamic_bias_option}"
                )

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(weight_params, logits)


class DynamicBiasSelector(Module):
    def __init__(
        self,
        cfg: "ModelConfig",
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = config
        self.main_config = cfg
        self.bias_option = self.cfg.bias_option
        self.model = self.__init_bias_model()

    def __init_bias_model(
        self,
    ) -> BiasHandlerAbstract:
        match self.bias_option:
            case DynamicBiasOptions.DEFAULT:
                return DefaultBiasHandler(self.main_config)
            case DynamicBiasOptions.SCALE_AND_OFFSET:
                return AffineBiasTransformHandler(self.main_config)
            case DynamicBiasOptions.ELEMENT_WISE_OFFSET:
                return ElementwiseBiasHandler(self.main_config)
            case DynamicBiasOptions.DYNAMIC_PARAMETERS:
                return BiasGeneratorHandler(self.main_config)
            case _:
                raise ValueError(f"Unsupported `bias_option`: {self.bias_option}")

    def forward(
        self,
        bias_params: Tensor,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(bias_params, logits)


# Old implementation
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
#         diagonal_model = LinearLayerStack(cfg).build_model()
#         anti_diagonal_model = None
#         if self.anti_diagonal_flag:
#             anti_diagonal_model = LinearLayerStack(cfg).build_model()
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
