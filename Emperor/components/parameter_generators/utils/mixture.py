import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from enum import Enum
import torch.nn.functional as F
from Emperor.base.utils import Module, DataClassBase, arange, reshape
from dataclasses import dataclass, field

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class MixtureConfig(DataClassBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model input dimension"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model output dimension"},
    )
    depth_dim: int | None = field(
        default=None,
        metadata={"help": "Mixture model depth dimension"},
    )
    top_k: int | None = field(
        default=None,
        metadata={
            "help": "Inidicates the top-k probs and indices to be selected from a distribution"
        },
    )
    weighted_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` the sepected parameters will be multiplied by their probs"
        },
    )
    bias_parameters_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` parameter `bias_parameters` will aslo be generated with the `weight_parameters`"
        },
    )
    num_experts: int | None = field(
        default=None,
        metadata={"help": "Router output dimension"},
    )
    dynamic_diagonal_params_flag: bool | None = field(
        default=None,
        metadata={
            "help": "Used for `GeneratorMixture`. When `True` `anti diagonal parameters` are added to every set or parameters for the topk samplers. Diagonal parameters are added by default"
        },
    )


class OuterProductNormOptions(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    LAYER_NORM = 4


class MixtureBase(Module):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "mixture_model_config", cfg)
        self.cfg: "MixtureConfig" = self._overwrite_config(config, overrides)

        self.input_dim = self.cfg.input_dim
        self.depth_dim = self.cfg.depth_dim
        self.output_dim = self.cfg.output_dim
        self.top_k = self.cfg.top_k
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag
        self.bias_parameters_flag = self.cfg.bias_parameters_flag
        self.num_experts = self.cfg.num_experts
        self.dynamic_diagonal_params_flag = self.cfg.dynamic_diagonal_params_flag
        assert self.depth_dim == self.num_experts, (
            "The `depth_dim` needs to be equal with `num_experts` since this is the dimension the router creates a distribution over."
        )
        if self.depth_dim == self.top_k:
            assert self.weighted_parameters_flag is True, (
                "If `full_mixture` is performed the `weighted_parameters_flag` must be True. Because the `weight_bank` or `bias_bank` needs to be broadcasted across the batch."
            )


class ParameterMixture(MixtureBase):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def compute_mixture(
        self,
        weight_probs: Tensor,
        weight_indices: Tensor | None = None,
        bias_probs: Tensor | None = None,
        bias_indices: Tensor | None = None,
        *args,
    ) -> tuple[Tensor, Tensor | None]:
        if self.top_k == 1:
            return self._compute_mixture_sparse(
                weight_probs,
                weight_indices,
                bias_probs,
                bias_indices,
                *args,
            )
        elif self.top_k == self.depth_dim:
            return self._compute_mixture_full(
                weight_probs,
                bias_probs,
                *args,
            )
        else:
            return self._compute_mixture_topk(
                weight_probs,
                weight_indices,
                bias_probs,
                bias_indices,
                *args,
            )

    def _compute_mixture_sparse(
        self,
        weight_probs: Tensor,
        weight_indices: Tensor,
        bias_probs: Tensor | None = None,
        bias_indices: Tensor | None = None,
        *args,
    ) -> tuple[Tensor, Tensor | None]:
        selected_params = self._select_parameters(weight_indices, bias_indices)

        weight_mixture, bias_mixture = self._compute_parameter_mixture(
            *selected_params,
            weight_probs,
            bias_probs,
            *args,
        )

        return weight_mixture, bias_mixture

    def _compute_mixture_topk(
        self,
        weight_probs: Tensor,
        weight_indices: Tensor,
        bias_probs: Tensor | None = None,
        bias_indices: Tensor | None = None,
        *args,
    ) -> tuple[Tensor, Tensor | None]:
        selected_params = self._select_parameters(weight_indices, bias_indices)
        weight_mixture, bias_mixture = self._compute_parameter_mixture(
            *selected_params,
            weight_probs,
            bias_probs,
            *args,
        )
        return weight_mixture, bias_mixture

    def _compute_mixture_full(
        self,
        weight_probs: Tensor,
        bias_probs: Tensor | None = None,
        *args,
    ) -> tuple[Tensor, Tensor | None]:
        weight_mixture, bias_mixture = self._compute_parameter_mixture(
            self.weight_bank,
            self.bias_bank,
            weight_probs,
            bias_probs,
            *args,
        )

        return weight_mixture, bias_mixture

    def _compute_parameter_mixture(
        self,
        selected_weight_parameters: Tensor,
        selected_bias_parameters: Tensor | None = None,
        weight_probs: Tensor | None = None,
        bias_probs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weight_mixture = self._compute_mixture(
            selected_weight_parameters,
            weight_probs,
        )
        bias_mixture = None
        if self.bias_parameters_flag:
            bias_mixture = self._compute_mixture(
                selected_bias_parameters,
                bias_probs,
                is_weight=False,
            )

        return weight_mixture, bias_mixture


class VectorMixture(ParameterMixture):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        config = getattr(cfg, "mixture_model_config", cfg)
        self.mixture_config: "MixtureConfig" = self._overwrite_config(config, overrides)
        self.weight_bank, self.bias_bank = self.__init_parameter_banks()

        range_weights, range_biases = self.__init_parameter_choice_ranges()
        self.register_buffer("range_weights", range_weights)
        self.register_buffer("range_biases", range_biases)

    def __init_parameter_banks(self) -> Tuple[Parameter, Parameter | None]:
        weight_bank_shape = (self.input_dim, self.depth_dim, self.output_dim)
        weight_bank = self._init_parameter_bank(weight_bank_shape)

        bias_bank = None
        if self.bias_parameters_flag:
            bias_bank_shape = (self.output_dim, self.depth_dim)
            bias_bank = self._init_parameter_bank(bias_bank_shape)

        return weight_bank, bias_bank

    def __init_parameter_choice_ranges(self) -> Tuple[Tensor, Tensor]:
        input_range = arange(self.input_dim)
        output_range = arange(self.output_dim)

        range_weight_shape = [1, self.input_dim]
        range_bias_shape = [1, self.output_dim]
        if 1 < self.top_k < self.depth_dim:
            range_weight_shape = [1, self.input_dim, 1]
            range_bias_shape = [1, self.output_dim, 1]

        choice_range_weights = reshape(input_range, range_weight_shape)
        choice_range_biases = reshape(output_range, range_bias_shape)

        return choice_range_weights, choice_range_biases

    def _select_parameters(
        self, weight_indices: Tensor, bias_indices: Tensor | None = None
    ) -> Tuple:
        selected_weights = self.__select_parameter_vectors(
            weight_indices, self.weight_bank, self.range_weights
        )
        selected_biases = None
        if self.bias_parameters_flag:
            selected_biases = self.__select_parameter_vectors(
                bias_indices, self.bias_bank, self.range_biases
            )

        return selected_weights, selected_biases

    def __select_parameter_vectors(
        self, indices: Tensor, weight_bank: Parameter, choice_range: Tensor
    ) -> Tensor:
        transposed_indices = indices.transpose(1, 0)
        return weight_bank[choice_range, transposed_indices]

    def _compute_mixture(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
        is_weight: bool = True,
    ) -> Tensor:
        parameter_mixture = self.__compute_weighted_parameters(
            selected_parameters, probs, is_weight
        )

        if self.top_k == 1:
            return parameter_mixture

        dim = -2 if is_weight else -1
        return parameter_mixture.sum(dim=dim)

    def __compute_weighted_parameters(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
        is_weight: bool = True,
    ):
        if self.weighted_parameters_flag:
            probs = probs.transpose(1, 0)
            if is_weight:
                probs = probs.unsqueeze(-1)
            return selected_parameters * probs
        return selected_parameters


class MatrixMixture(ParameterMixture):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        config = getattr(cfg, "mixture_model_config", cfg)
        self.mixture_config: "MixtureConfig" = self._overwrite_config(config, overrides)
        self.weight_bank, self.bias_bank = self.__init_parameter_banks()

        self.weight_probs_shape, self.bias_probs_shape = (
            self.__generate_probability_shapes()
        )

        if self.depth_dim == self.top_k:
            assert self.weighted_parameters_flag is True

    def __generate_probability_shapes(self) -> Tuple:
        weight_probs_shape = (-1, self.top_k, 1)
        bias_probs_shape = (-1, self.top_k)
        if self.top_k > 1:
            weight_probs_shape = (-1, self.top_k, 1, 1)
            bias_probs_shape = (-1, self.top_k, 1)
        return weight_probs_shape, bias_probs_shape

    def __init_parameter_banks(self) -> Tuple[Parameter, Parameter | None]:
        weight_bank_shape = (self.depth_dim, self.input_dim, self.output_dim)
        weight_bank = self._init_parameter_bank(weight_bank_shape)

        bias_bank = None
        if self.bias_parameters_flag:
            bias_bank_shape = (self.depth_dim, self.output_dim)
            bias_bank = self._init_parameter_bank(bias_bank_shape)

        return weight_bank, bias_bank

    def _select_parameters(
        self,
        weight_indices: Tensor,
        bias_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        selected_weights = self.weight_bank[weight_indices]
        selected_biases = None
        if self.bias_parameters_flag:
            selected_biases = self.bias_bank[bias_indices]

        return selected_weights, selected_biases

    def _compute_mixture(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
        is_weight: bool = True,
    ) -> Tensor:
        parameter_mixture = self.__compute_weighted_parameters(
            selected_parameters, probs, is_weight
        )

        if self.top_k > 1:
            return parameter_mixture.sum(dim=1)
        return parameter_mixture

    def __compute_weighted_parameters(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
        is_weight: bool = True,
    ) -> Tensor:
        probs_shape = self.bias_probs_shape
        if is_weight:
            probs_shape = self.weight_probs_shape

        if self.weighted_parameters_flag:
            probs = reshape(probs, probs_shape)
            return selected_parameters * probs
        return selected_parameters


class GeneratorMixture(ParameterMixture):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        config = getattr(cfg, "mixture_model_config", cfg)
        self.mixture_config: "MixtureConfig" = self._overwrite_config(config, overrides)
        self.diagonal_dim = min(self.input_dim, self.output_dim)

        self.einsum_vector_operation = self.__decide_einsum_computation()
        self.diagonal_padding_shape = self.__compute_diagonal_shape()
        self.weight_probs_shape, self.bias_probs_shape = (
            self.__generate_probability_shapes()
        )

        (
            self.input_weight_bank,
            self.output_weight_bank,
            self.diagonal_weight_bank,
            self.anti_diagonal_weight_bank,
            self.bias_bank,
        ) = self.__init_parameter_banks()

        self.outper_product_norm = nn.LayerNorm(
            self.output_dim,
            elementwise_affine=False,
            # bias=False,
        )

    def __decide_einsum_computation(self) -> str:
        if self.top_k == self.num_experts:
            return "bi,kij->bkj"
        return "bi,bkij->bkj"

    def __compute_diagonal_shape(self) -> tuple | None:
        diagonal_padding_shape = None
        if self.input_dim != self.output_dim:
            padding_size = abs(self.input_dim - self.output_dim)
            diagonal_padding_shape = (0, padding_size, 0, 0)
            if self.input_dim > self.output_dim:
                diagonal_padding_shape = (0, 0, 0, padding_size)
        return diagonal_padding_shape

    def __generate_probability_shapes(self) -> tuple[tuple, tuple]:
        weight_probs_shape = (-1, self.top_k, 1, 1)
        bias_probs_shape = (-1, self.top_k, 1)
        return weight_probs_shape, bias_probs_shape

    def __init_parameter_banks(
        self,
    ) -> Tuple[Parameter, Parameter, Parameter, Parameter | None, Parameter | None]:
        input_weight_shape = (self.depth_dim, self.input_dim, self.input_dim)
        output_weight_shape = (self.depth_dim, self.input_dim, self.output_dim)
        diagonal_weight_shape = (self.depth_dim, self.input_dim, self.diagonal_dim)
        anti_diagonal_weight_shape = (self.depth_dim, self.input_dim, self.diagonal_dim)
        bias_bank_shape = (self.depth_dim, self.input_dim, self.output_dim)

        input_weight_bank = self._init_parameter_bank(input_weight_shape)
        output_weight_bank = self._init_parameter_bank(output_weight_shape)
        diagonal_weight_bank = self._init_parameter_bank(diagonal_weight_shape)

        anti_diagonal_weight_bank = None
        if self.dynamic_diagonal_params_flag:
            anti_diagonal_weight_bank = self._init_parameter_bank(
                anti_diagonal_weight_shape
            )

        bias_bank = None
        if self.bias_parameters_flag:
            bias_bank = self._init_parameter_bank(bias_bank_shape)

        return (
            input_weight_bank,
            output_weight_bank,
            diagonal_weight_bank,
            anti_diagonal_weight_bank,
            bias_bank,
        )

    def _select_parameters(
        self,
        weight_indices: Tensor,
        bias_indices: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.top_k == 1:
            weight_indices = weight_indices.unsqueeze(dim=-1)
            if self.bias_parameters_flag:
                bias_indices = bias_indices.unsqueeze(dim=-1)

        selected_input_params = self.input_weight_bank[weight_indices]
        selected_output_params = self.output_weight_bank[weight_indices]
        selected_diagonal_params = self.diagonal_weight_bank[weight_indices]

        selected_anti_diagonal_params = None
        if self.dynamic_diagonal_params_flag:
            selected_anti_diagonal_params = self.anti_diagonal_weight_bank[
                weight_indices
            ]

        selected_bias_params = None
        if self.bias_parameters_flag:
            selected_bias_params = self.bias_bank[bias_indices]

        return (
            selected_input_params,
            selected_output_params,
            selected_diagonal_params,
            selected_anti_diagonal_params,
            selected_bias_params,
        )

    def _compute_mixture_full(
        self,
        weight_probs: Tensor,
        bias_probs: Tensor | None = None,
        *args,
    ) -> tuple[Tensor, Tensor | None]:
        weight_mixture, bias_mixture = self._compute_parameter_mixture(
            self.input_weight_bank,
            self.output_weight_bank,
            self.diagonal_weight_bank,
            self.anti_diagonal_weight_bank,
            self.bias_bank,
            weight_probs,
            bias_probs,
            *args,
        )

        return weight_mixture, bias_mixture

    def _compute_parameter_mixture(
        self,
        input_weight_params: Tensor,
        output_weight_params: Tensor,
        diagonal_params: Tensor,
        anti_diagonal_params: Tensor | None = None,
        bias_params: Tensor | None = None,
        weight_probs: Tensor | None = None,
        bias_probs: Tensor | None = None,
        input_batch: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor | None]:
        (
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            bias_vectors,
        ) = self.__compute_parameter_vectors(
            input_batch,
            input_weight_params,
            output_weight_params,
            diagonal_params,
            anti_diagonal_params,
            bias_params,
        )

        generated_weights = self.__generate_weight_parameters(
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            weight_probs,
        )

        generated_biases = self.__generate_bias_parameters(bias_vectors, bias_probs)

        return generated_weights, generated_biases

    def __compute_parameter_vectors(
        self,
        input_batch: Tensor,
        input_weight_params: Tensor,
        output_weight_params: Tensor,
        diagonal_weight_params: Tensor,
        anti_diagonal_params: Tensor | None = None,
        bias_params: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        input_vectors = self.__compute_einsum(input_batch, input_weight_params)
        output_vectors = self.__compute_einsum(input_batch, output_weight_params)
        diagonal_vectors = self.__compute_einsum(input_batch, diagonal_weight_params)
        anti_diagonal_vectors = self.__maybe_compute_einsum(
            input_batch, anti_diagonal_params, self.dynamic_diagonal_params_flag
        )
        bias_output = self.__maybe_compute_einsum(
            input_batch, bias_params, self.bias_parameters_flag
        )

        return (
            input_vectors,
            output_vectors,
            diagonal_vectors,
            anti_diagonal_vectors,
            bias_output,
        )

    def __maybe_compute_einsum(
        self, input_batch: Tensor, weight_params: Tensor, einsum_flag: bool = False
    ) -> Tensor | None:
        if einsum_flag:
            return self.__compute_einsum(input_batch, weight_params)
        return None

    def __compute_einsum(
        self,
        input_batch: Tensor | None,
        weight_params: Tensor,
    ) -> Tensor:
        if input_batch is None:
            raise RuntimeError(
                "`compute_mixture` requires an `input_batch` argument when using `GeneratorMixture`."
            )
        vectors = torch.einsum(self.einsum_vector_operation, input_batch, weight_params)
        return vectors

    def __generate_weight_parameters(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
        diagonal_vectors: Tensor,
        anti_diagonal_vectors: Tensor | None = None,
        weight_probs: Tensor | None = None,
    ) -> Tensor:
        outer_product = self.__compute_outer_product(input_vectors, output_vectors)
        diagonal_matrix = self.__compute_diagonal_matrix(diagonal_vectors)
        anti_diagonal_matrix = self.__compute_anti_diagonal_matrix(
            anti_diagonal_vectors
        )
        generated_parameters = self.__assemble_parameters_matrix(
            outer_product, diagonal_matrix, anti_diagonal_matrix
        )

        weighted_parameters = self.__apply_parameter_weighting(
            generated_parameters, self.weight_probs_shape, weight_probs
        )

        if self.top_k > 1:
            return torch.sum(weighted_parameters, dim=1)
        return weighted_parameters.squeeze(1)

    def __compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        normalize_before = True
        if normalize_before:
            input_vectors = self.__normalize_outer_product_parameters(
                input_vectors,
                OuterProductNormOptions.TANH,
            )
            output_vectors = self.__normalize_outer_product_parameters(
                output_vectors,
                OuterProductNormOptions.TANH,
            )

        outer_product = torch.einsum("bij,bik->bijk", input_vectors, output_vectors)

        if normalize_before:
            return outer_product

        return self.__normalize_outer_product_parameters(
            outer_product, OuterProductNormOptions.TANH
        )

    def __normalize_outer_product_parameters(
        self,
        outer_product: Tensor,
        outer_product_norm_option: OuterProductNormOptions | None,
    ) -> Tensor:
        match outer_product_norm_option:
            case OuterProductNormOptions.RELU:
                return F.relu(outer_product)
            case OuterProductNormOptions.TANH:
                return F.tanh(outer_product)
            case OuterProductNormOptions.SIGMOID:
                return F.sigmoid(outer_product)
            case OuterProductNormOptions.LAYER_NORM:
                # TODO: Layer usualy has a scalar and bias that is applied
                # to the normalized output. In this case i need in the future
                # to select the scalar and bias based on the parameters
                # chosen by the router and sampler
                return self.outper_product_norm(outer_product)
            case _:
                return outer_product

    def __compute_anti_diagonal_matrix(
        self,
        anti_diagonal_vectors: Tensor | None = None,
    ) -> Tensor | None:
        if self.dynamic_diagonal_params_flag:
            anti_diagonal_matrix = self.__compute_diagonal_matrix(anti_diagonal_vectors)
            return anti_diagonal_matrix.flip(dims=[2])
        return None

    def __compute_diagonal_matrix(
        self,
        diagonal_vectors: Tensor,
    ) -> Tensor:
        diagonal_matrix = torch.diag_embed(diagonal_vectors)
        if self.diagonal_padding_shape is not None:
            diagonal_matrix = F.pad(diagonal_matrix, self.diagonal_padding_shape)
        return diagonal_matrix

    def __assemble_parameters_matrix(
        self,
        outer_product: Tensor,
        diagonal_matrix: Tensor,
        anti_diagonal_matrix: Tensor | None = None,
    ) -> Tensor:
        generated_weights = outer_product + diagonal_matrix
        if self.dynamic_diagonal_params_flag:
            generated_weights = generated_weights + anti_diagonal_matrix
        return generated_weights

    def __apply_parameter_weighting(
        self,
        generated_parameters: Tensor,
        parameter_shape: tuple,
        weight_probs: Tensor | None = None,
    ) -> Tensor:
        if self.weighted_parameters_flag:
            weight_probs = reshape(weight_probs, parameter_shape)
            return generated_parameters * weight_probs
        return generated_parameters

    def __generate_bias_parameters(
        self,
        generated_biases: Tensor | None = None,
        bias_probs: Tensor | None = None,
    ) -> Tensor | None:
        if not self.bias_parameters_flag:
            return None

        weighted_biases = self.__apply_parameter_weighting(
            generated_biases, self.bias_probs_shape, bias_probs
        )

        if self.top_k > 1:
            return torch.sum(weighted_biases, dim=1)
        return weighted_biases.squeeze(1)
