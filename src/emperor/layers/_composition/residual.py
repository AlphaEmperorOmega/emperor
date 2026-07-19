import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.layers._config import ResidualConfig
from emperor.layers._options import ResidualConnectionOptions
from emperor.layers._validation import ResidualConnectionValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.linears import LinearLayer, LinearLayerConfig


class ResidualConnection(Module):
    VALIDATOR = ResidualConnectionValidator
    WEIGHTED_BLEND_INITIAL_ALPHA = 0.9

    def __init__(
        self,
        cfg: ResidualConfig,
        overrides: ResidualConfig | None = None,
    ):
        super().__init__()
        self.cfg: ResidualConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.option: ResidualConnectionOptions = self.cfg.option
        self.residual_dim: int | None = self.cfg.residual_dim
        self.model_config: LinearLayerConfig | None = self.cfg.model_config
        self.raw_weight, self.model = self.__initialize_residual_components()
        self.__initialize_data_dependent_coefficient_model()

    def __initialize_residual_components(
        self,
    ) -> "tuple[nn.Parameter | None, LinearLayer | None]":
        if self.option == ResidualConnectionOptions.RESIDUAL:
            return None, None
        if self.model_config is not None:
            coefficient_model = self._build_from_config(
                self.model_config,
                input_dim=self.residual_dim * 2,
                output_dim=self.residual_dim,
            )
            return None, coefficient_model

        raw_mix_coefficient = self.__initial_raw_mix_coefficient()
        return nn.Parameter(raw_mix_coefficient), None

    def __initial_raw_mix_coefficient(self) -> Tensor:
        match self.option:
            case ResidualConnectionOptions.WEIGHTED_RESIDUAL:
                return torch.tensor(0.0)
            case ResidualConnectionOptions.WEIGHTED_BLEND:
                alpha = self.WEIGHTED_BLEND_INITIAL_ALPHA
                return torch.tensor(math.log(alpha / (1.0 - alpha)))
            case _:
                raise ValueError(
                    f"Residual option does not use mixing coefficients: {self.option}."
                )

    def __initialize_data_dependent_coefficient_model(self) -> None:
        if self.model is None:
            return

        initial_raw_mix_coefficient = self.__initial_raw_mix_coefficient().item()
        nn.init.zeros_(self.model.weight_params)
        nn.init.constant_(self.model.bias_params, initial_raw_mix_coefficient)

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        if self.option == ResidualConnectionOptions.RESIDUAL:
            return current + previous
        if self.option == ResidualConnectionOptions.WEIGHTED_RESIDUAL:
            return self.__apply_weighted_residual(current, previous)
        if self.option == ResidualConnectionOptions.WEIGHTED_BLEND:
            return self.__apply_weighted_blend(current, previous)
        raise ValueError(
            "Unsupported residual connection option "
            f"{self.option} for ResidualConnection."
        )

    def __apply_weighted_residual(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        raw_mix_coefficient = self.__resolve_raw_mix_coefficient(current, previous)
        residual_weight = torch.tanh(raw_mix_coefficient)
        return previous + residual_weight * current

    def __apply_weighted_blend(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        raw_mix_coefficient = self.__resolve_raw_mix_coefficient(current, previous)
        current_blend_coefficient = torch.sigmoid(raw_mix_coefficient)
        previous_blend_coefficient = 1.0 - current_blend_coefficient
        current_blend_contribution = current_blend_coefficient * current
        previous_blend_contribution = previous_blend_coefficient * previous
        return current_blend_contribution + previous_blend_contribution

    def __resolve_raw_mix_coefficient(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        if self.model is not None:
            coefficient_model_input = torch.cat((current, previous), dim=-1)
            return self.model(coefficient_model_input)
        return self.VALIDATOR.validate_raw_mix_coefficient(
            self.raw_weight,
            self.option,
        )
