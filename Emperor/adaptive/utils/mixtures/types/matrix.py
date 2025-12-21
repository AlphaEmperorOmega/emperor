from torch import Tensor
from Emperor.adaptive.utils.mixtures.base import MixtureBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.adaptive.utils.mixture import MixtureConfig


class MatrixMixtureBase(MixtureBase):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)

    def compute_mixture(
        self,
        probs: Tensor,
        indices: Tensor | None = None,
    ) -> Tensor:
        selected_params = self._select_parameters(indices)
        return self.__compute_parameter_mixture(selected_params, probs)

    def _select_parameters(
        self,
        indices: Tensor | None = None,
    ) -> Tensor:
        if indices is None:
            return self.parameter_bank
        return self.parameter_bank[indices]

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

        if self.__is_topk_sparse():
            return weighted_parameters
        return weighted_parameters.sum(dim=1)

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
        probs = probs.reshape(self.probability_shape)
        return selected_parameters * probs


class MatrixWeightsMixture(MatrixMixtureBase):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.parameter_mixture_dim = -2
        self.probability_shape = self._generate_probability_shapes()
        self.parameter_bank_shape = (self.depth_dim, self.input_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.register_buffer("select_range", self._init_parameter_select_range())

    def _generate_probability_shapes(self) -> tuple:
        if self.top_k > 1:
            return (-1, self.top_k, 1, 1)
        return (-1, self.top_k, 1)


class MatrixBiasMixture(MatrixMixtureBase):
    def __init__(
        self,
        cfg: "MixtureConfig | ModelConfig",
        overrides: "MixtureConfig | None" = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.parameter_mixture_dim = -1
        self.parameter_bank_shape = (self.depth_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
        self.register_buffer("select_range", self._init_parameter_select_range())

    def _generate_probability_shapes(self) -> tuple:
        if self.top_k > 1:
            return (-1, self.top_k, 1)
        return (-1, self.top_k)
