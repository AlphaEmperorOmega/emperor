import torch
from enum import Enum
from torch import Tensor
from torch.nn import functional as F
from Emperor.adaptive.utils.mixture import AdaptiveMixtureConfig
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase
from Emperor.experts.utils.layers import MixtureOfExperts


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class OuterProductNormOptions(Enum):
    RELU = 1
    TANH = 2
    SIGMOID = 3
    LAYER_NORM = 4


class GeneratorMixtureBase(AdaptiveMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        config = getattr(cfg, "mixture_model_config", cfg)
        self.mixture_config: "AdaptiveMixtureConfig" = self._overwrite_config(
            config, overrides
        )

    def _is_topk_sparse(self) -> bool:
        return self.top_k == 1


class GeneratorWeightsMixture(GeneratorMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.range_dim = self.input_dim
        self.parameter_mixture_dim = -2
        self.probability_shape = (-1, self.top_k, 1, 1)
        self.input_vector_generator = MixtureOfExperts(self.mixture_of_experts_config)
        self.output_vector_generator = MixtureOfExperts(self.mixture_of_experts_config)
        self.register_buffer("select_range", self._init_parameter_select_range())

    def compute_mixture(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor | None = None,
    ) -> Tensor:
        inputs = (input_batch, indices, probabilities)
        input_vectors = self.input_vector_generator(*inputs)
        output_vectors = self.output_vector_generator(*inputs)
        generated_parameters = self.__compute_outer_product(
            input_vectors, output_vectors
        )
        return self.__compute_parameter_mixture(generated_parameters, probabilities)

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

    def __compute_parameter_mixture(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
    ) -> Tensor:
        weighted_parameters = selected_parameters
        if self.__should_compute_weighted_parameters(probs):
            weighted_parameters = self.__apply_parameter_weighting(
                selected_parameters, self.probability_shape, probs
            )

        if self.__is_topk_sparse():
            return weighted_parameters.squeeze(1)
        return torch.sum(weighted_parameters, dim=1)

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

    def __apply_parameter_weighting(
        self,
        generated_parameters: Tensor,
        parameter_shape: tuple,
        probs: Tensor | None = None,
    ) -> Tensor:
        if self.__should_compute_weighted_parameters(probs):
            weight_probs = probs.reshape(parameter_shape)
            return generated_parameters * weight_probs
        return generated_parameters

    def __should_compute_weighted_parameters(self, probs: Tensor | None) -> bool | None:
        if self.weighted_parameters_flag and probs is None:
            raise ValueError(
                "Probabilities must be provided when 'weighted_parameters_flag' is set to True."
            )
        return self.weighted_parameters_flag and probs is not None


class GeneratorBiasMixture(GeneratorMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.range_dim = self.output_dim
        self.parameter_mixture_dim = -1
        self.probability_shape = (-1, self.top_k, 1)
        self.bias_generator = MixtureOfExperts(cfg)
        self.register_buffer("select_range", self._init_parameter_select_range())

    def compute_mixture(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor | None = None,
    ) -> Tensor:
        inputs = (input_batch, indices, probabilities)
        return self.bias_generator(*inputs)
