"""Matrix banks consumed by the parametric layer's routing pipeline."""

from dataclasses import replace

from torch import Tensor

from emperor.nn import Module
from emperor.parametric._mixtures.config import (
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)
from emperor.parametric._mixtures.validation import MatrixMixtureValidator


class MatrixMixtureBase(Module):
    VALIDATOR = MatrixMixtureValidator

    def __init__(self, cfg, overrides=None):
        super().__init__()
        config = self._override_config(cfg, overrides)
        self.VALIDATOR.validate_configuration(config)
        self.cfg = replace(
            config,
            weighted_parameters_flag=True
            if config.weighted_parameters_flag is None
            else config.weighted_parameters_flag,
        )
        self.VALIDATOR.validate_field_types(self.cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.num_experts = self.cfg.num_experts
        self.top_k = self.cfg.top_k
        self.weighted_parameters_flag = self.cfg.weighted_parameters_flag

    def compute_mixture(
        self, probabilities: Tensor | None, indices: Tensor | None = None, *args
    ) -> Tensor:
        self.VALIDATOR.validate_route(self, probabilities, indices)
        if indices is None:
            if not self.weighted_parameters_flag:
                return self.parameter_bank.sum(dim=0)
            selected = self.parameter_bank.unsqueeze(0)
        else:
            selected = self.parameter_bank[indices.reshape(-1, self.top_k)]
        if self.weighted_parameters_flag:
            probability_shape = (-1, self.top_k) + (1,) * (self.parameter_bank.ndim - 1)
            selected = selected * probabilities.reshape(probability_shape)
        return selected.sum(dim=1)


class MatrixWeightsMixture(MatrixMixtureBase):
    def __init__(
        self,
        cfg: MatrixWeightsMixtureConfig,
        overrides: MatrixWeightsMixtureConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.parameter_bank_shape = (self.num_experts, self.input_dim, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)


class MatrixBiasMixture(MatrixMixtureBase):
    def __init__(
        self,
        cfg: MatrixBiasMixtureConfig,
        overrides: MatrixBiasMixtureConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.parameter_bank_shape = (self.num_experts, self.output_dim)
        self.parameter_bank = self._init_parameter_bank(self.parameter_bank_shape)
