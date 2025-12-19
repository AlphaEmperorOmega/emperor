import torch
from enum import Enum
from torch import Tensor
from torch.nn import functional as F
from Emperor.adaptive.utils.mixture import AdaptiveMixtureConfig
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase
from Emperor.experts.utils.layers import MixtureOfExperts, MixtureOfExpertsConfig


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
        self.input_vector_generator, self.output_vector_generator = (
            self.__init_generators()
        )

    def __init_generators(self):
        options = {}
        if self.weighted_parameters_flag:
            options = {"weighted_parameters_flag": False}

        input_overrides = MixtureOfExpertsConfig(
            output_dim=self.input_dim,
            compute_expert_mixture_flag=False,
            **options,
        )
        output_overrides = MixtureOfExpertsConfig(
            output_dim=self.output_dim,
            compute_expert_mixture_flag=False,
            **options,
        )
        input_vector_generator = MixtureOfExperts(self.main_cfg, input_overrides)
        output_vector_generator = MixtureOfExperts(self.main_cfg, output_overrides)
        return input_vector_generator, output_vector_generator

    def compute_mixture(
        self,
        input_batch: Tensor,
        probabilities: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> tuple[Tensor | Tensor]:
        experts_inputs = (input_batch, probabilities, indices)
        input_vectors, input_loss = self.input_vector_generator(*experts_inputs)
        output_vectors, output_loss = self.output_vector_generator(*experts_inputs)
        generated_parameters = self.__compute_outer_product(
            input_vectors, output_vectors
        )
        parameter_mixture = self.__compute_parameter_mixture(
            generated_parameters, probabilities
        )
        total_loss = input_loss + output_loss
        return parameter_mixture, total_loss

    def __compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        normalize_before = True
        if normalize_before:
            input_vectors = self.__normalize_outer_product_parameters(input_vectors)
            output_vectors = self.__normalize_outer_product_parameters(output_vectors)

        outer_product = torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

        if normalize_before:
            return outer_product

        return self.__normalize_outer_product_parameters(outer_product)

    def __compute_parameter_mixture(
        self,
        selected_parameters: Tensor,
        probabilities: Tensor,
    ) -> Tensor:
        weighted_parameters = selected_parameters
        if self.__should_compute_weighted_parameters(probabilities):
            weighted_parameters = self.__apply_parameter_weighting(
                selected_parameters, self.probability_shape, probabilities
            )

        if self.__is_topk_sparse():
            return weighted_parameters.squeeze(1)
        return torch.sum(weighted_parameters, dim=1)

    def __is_topk_sparse(self) -> bool:
        return self.top_k == 1

    def __normalize_outer_product_parameters(
        self,
        outer_product: Tensor,
    ) -> Tensor:
        return torch.clamp(outer_product, -5.0, 5.0)

    def __apply_parameter_weighting(
        self,
        generated_parameters: Tensor,
        parameter_shape: tuple,
        probs: Tensor | None = None,
    ) -> Tensor:
        weight_probs = probs.reshape(parameter_shape)
        return generated_parameters * weight_probs

    def __should_compute_weighted_parameters(
        self, probabilities: Tensor | None = None
    ) -> bool | None:
        if self.weighted_parameters_flag and probabilities is None:
            raise ValueError(
                "Probabilities must be provided when 'weighted_parameters_flag' is set to True."
            )
        return self.weighted_parameters_flag and probabilities is not None


class GeneratorBiasMixture(GeneratorMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.range_dim = self.output_dim
        output_overrides = MixtureOfExpertsConfig(
            output_dim=self.output_dim,
            compute_expert_mixture_flag=False,
            weighted_parameters_flag=True,
        )
        self.bias_generator = MixtureOfExperts(self.main_cfg, output_overrides)

    def compute_mixture(
        self,
        input_batch: Tensor,
        probabilities: Tensor,
        indices: Tensor | None = None,
    ) -> Tensor:
        inputs = (input_batch, indices, probabilities)
        return self.bias_generator(*inputs)
