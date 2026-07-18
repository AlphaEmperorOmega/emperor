from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.experts import MixtureOfExpertsConfig, RoutingInitializationMode
from emperor.parametric._mixtures.base import AdaptiveMixtureBase
from emperor.parametric._mixtures.config import (
    AdaptiveMixtureConfig,
    ClipParameterOptions,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class GeneratorMixtureBase(AdaptiveMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.generator_config = self.cfg.generator_config

    def _is_topk_sparse(self) -> bool:
        return self.top_k == 1

    def _build_generator(self, output_dim: int, **kwargs):
        overrides = MixtureOfExpertsConfig(
            input_dim=self.input_dim,
            output_dim=output_dim,
            top_k=self.top_k,
            num_experts=self.num_experts,
            **kwargs,
        )
        return self.generator_config.build(overrides)


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
        options = {
            "compute_expert_mixture_flag": False,
        }
        if self.weighted_parameters_flag:
            options = {
                "compute_expert_mixture_flag": False,
                "weighted_parameters_flag": False,
                "routing_initialization_mode": RoutingInitializationMode.DISABLED,
            }
        input_vector_generator = self._build_generator(self.input_dim, **options)
        output_vector_generator = self._build_generator(self.output_dim, **options)
        return input_vector_generator, output_vector_generator

    def compute_mixture(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_input_batch_2d(input_batch)
        self.VALIDATOR.validate_weighted_probabilities(self.cfg, probabilities)
        experts_inputs = (input_batch, probabilities, indices)
        input_vectors, _, input_loss = self.input_vector_generator(*experts_inputs)
        output_vectors, _, output_loss = self.output_vector_generator(*experts_inputs)
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
        if self.clip_parameter_option == ClipParameterOptions.BEFORE:
            input_vectors = self.__normalize_parameters(input_vectors)
            output_vectors = self.__normalize_parameters(output_vectors)
        if input_vectors.dim() == 2:
            input_vectors = input_vectors.view(-1, self.top_k, self.input_dim)
            output_vectors = output_vectors.view(-1, self.top_k, self.output_dim)

        outer_product = torch.einsum("bki,bkj->bkij", input_vectors, output_vectors)

        if self.clip_parameter_option == ClipParameterOptions.AFTER:
            return self.__normalize_parameters(outer_product)
        return outer_product

    def __compute_parameter_mixture(
        self,
        selected_parameters: Tensor,
        probabilities: Tensor | None,
    ) -> Tensor:
        weighted_parameters = selected_parameters
        if self.__should_compute_weighted_parameters(probabilities):
            weighted_parameters = self.__apply_parameter_weighting(
                selected_parameters, probabilities
            )

        if self._is_topk_sparse():
            return weighted_parameters.squeeze(1)
        return torch.sum(weighted_parameters, dim=1)

    def __normalize_parameters(
        self,
        outer_product: Tensor,
    ) -> Tensor:
        return torch.clamp(outer_product, -self.clip_range, self.clip_range)

    def __apply_parameter_weighting(
        self,
        generated_parameters: Tensor,
        probabilities: Tensor | None = None,
    ) -> Tensor:
        probabilities = probabilities.view(self.probability_shape)
        return generated_parameters * probabilities

    def __should_compute_weighted_parameters(
        self, probabilities: Tensor | None = None
    ) -> bool | None:
        is_weight_flag = self.weighted_parameters_flag
        are_probabilities = probabilities is not None

        return is_weight_flag and are_probabilities


class GeneratorBiasMixture(GeneratorMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.range_dim = self.output_dim
        self.bias_generator = self.__init_generator()

    def __init_generator(self):
        return self._build_generator(
            self.output_dim,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
        )

    def compute_mixture(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_input_batch_2d(input_batch)
        bias, _, loss = self.bias_generator(input_batch, probabilities, indices)
        return bias, loss
