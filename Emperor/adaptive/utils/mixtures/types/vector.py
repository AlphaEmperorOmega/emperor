import torch

from torch import Tensor
from Emperor.adaptive.utils.mixtures.types.utils._validator import (
    _VectorMixtureValidator,
)
from Emperor.adaptive.utils.mixtures.base import (
    AdaptiveMixtureBase,
    AdaptiveMixtureConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class VectorMixtureBase(AdaptiveMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.depth_dim = self.num_experts

        self.validator = _VectorMixtureValidator(self)

    def _init_parameter_select_range(self) -> Tensor:
        input_range = torch.arange(self.range_dim)
        range_shape = [1, self.range_dim]
        if 1 < self.top_k < self.depth_dim:
            range_shape = [1, self.range_dim, 1]
        return input_range.reshape(range_shape)

    def compute_mixture(
        self, probabilities: Tensor, indices: Tensor | None = None, *args
    ) -> Tensor:
        selected_params = self._select_parameters(indices)
        return self.__compute_parameter_mixture(selected_params, probabilities)

    def _select_parameters(self, indices: Tensor | None) -> Tensor:
        if indices is None:
            return self.parameter_bank
        indices = indices.transpose(1, 0)
        return self.parameter_bank[self.select_range, indices]

    def __compute_parameter_mixture(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
    ) -> Tensor:
        weighted_parameters = selected_parameters

        if self.__should_compute_weighted_parameters(probs):
            weighted_parameters = self._compute_weighted_parameters(
                selected_parameters, probs
            )

        if not self.__is_topk_sparse():
            weighted_parameters = weighted_parameters.sum(
                dim=self.parameter_mixture_dim
            )
        return weighted_parameters

    def _handle_mixture_output(self, weighted_parameters: Tensor):
        return weighted_parameters

    def __is_topk_sparse(self) -> bool:
        return self.top_k == 1

    def __should_compute_weighted_parameters(self, probs: Tensor | None) -> bool | None:
        if self.weighted_parameters_flag and probs is None:
            raise ValueError(
                "Probabilities must be provided when 'weighted_parameters_flag' is set to True."
            )
        return self.weighted_parameters_flag and probs is not None

    def _compute_weighted_parameters(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "The method '_compute_weighted_parameters' must be implemented in the child class."
        )


class VectorWeightsMixture(VectorMixtureBase):
    def __init__(
        self,
        cfg: "AdaptiveMixtureConfig | ModelConfig",
        overrides: "AdaptiveMixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.range_dim = self.input_dim
        self.parameter_mixture_dim = -2
        self.parameter_bank_shape = (self.input_dim, self.depth_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.register_buffer("select_range", self._init_parameter_select_range())

    def _compute_weighted_parameters(
        self,
        selected_parameters: Tensor,
        probs: Tensor,
    ) -> Tensor:
        probs = probs.transpose(1, 0).unsqueeze(-1)
        return selected_parameters * probs

    def _handle_mixture_output(self, weighted_parameters: Tensor):
        return torch.transpose(weighted_parameters, -1, -2)


# TODO: Maybe in the future come up with a better idea to add biases
# for this type of model
# class VectorBiasMixture(VectorMixtureBase):
#     def __init__(
#         self,
#         cfg: "AdaptiveMixtureConfig | ModelConfig",
#         overrides: "AdaptiveMixtureConfig | None" = None,
#     ) -> None:
#         super().__init__(cfg, overrides)
#         self.range_dim = self.output_dim
#         self.parameter_mixture_dim = -1
#         self.parameter_bank_shape = (self.output_dim, self.depth_dim)
#         self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
#         self.register_buffer("select_range", self._init_parameter_select_range())
#
#     def _compute_weighted_parameters(
#         self,
#         selected_parameters: Tensor,
#         probs: Tensor,
#     ):
#         probs = probs.transpose(1, 0)
#         return selected_parameters * probs
