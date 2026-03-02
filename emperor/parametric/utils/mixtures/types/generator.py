import torch

from torch import Tensor
from emperor.experts.utils.enums import InitSamplerOptions
from emperor.experts.utils.layers import MixtureOfExperts, MixtureOfExpertsConfig
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.parametric.utils.mixtures.types.utils._validator import (
    _GeneratorMixtureValidator,
)
from emperor.parametric.utils.mixtures.base import (
    AdaptiveMixtureBase,
    AdaptiveMixtureConfig,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


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

        self.validator = _GeneratorMixtureValidator(self)

    def __init_generators(self):
        options = {
            "compute_expert_mixture_flag": False,
        }
        if self.weighted_parameters_flag:
            options = {
                "compute_expert_mixture_flag": False,
                "weighted_parameters_flag": False,
                "init_sampler_option": InitSamplerOptions.DISABLED,
            }
        input_overrides = MixtureOfExpertsConfig(
            input_dim=self.input_dim, output_dim=self.input_dim, **options
        )
        output_overrides = MixtureOfExpertsConfig(
            input_dim=self.input_dim, output_dim=self.output_dim, **options
        )
        input_vector_generator = MixtureOfExperts(self.main_cfg, input_overrides)
        output_vector_generator = MixtureOfExperts(self.main_cfg, output_overrides)
        return input_vector_generator, output_vector_generator

    def compute_mixture(
        self,
        probabilities: Tensor | None,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> tuple[Tensor | Tensor]:
        self.validator.ensure_input_batch_is_2D_tensor(input_batch)
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
        probabilities: Tensor,
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

        self.validator.ensure_mixture_weighted_flag_is_false()
        self.validator.ensure_probabilities_exist_for_weighted_flag(probabilities)

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
        output_overrides = MixtureOfExpertsConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            weighted_parameters_flag=True,
        )
        return MixtureOfExperts(self.main_cfg, output_overrides)

    def compute_mixture(
        self,
        probabilities: Tensor,
        indices: Tensor | None,
        input_batch: Tensor,
    ) -> Tensor:
        return self.bias_generator(input_batch, probabilities, indices)
