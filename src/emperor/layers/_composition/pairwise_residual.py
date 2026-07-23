from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from emperor.layers._composition.residual import ResidualConnection
    from emperor.linears import LinearLayer, LinearLayerConfig
    from emperor.nn import Module


@dataclass(frozen=True, slots=True)
class _PairwiseResidualParameters:
    raw_weight: nn.Parameter | None = None
    model: LinearLayer | None = None


class PairwiseResidual(ABC):
    """Construct coefficient parameters and compose two residual sources."""

    USES_MIX_COEFFICIENT: ClassVar[bool] = False

    @staticmethod
    def initial_raw_mix_coefficient(
        blend_initial_alpha: float,
    ) -> Tensor | None:
        return None

    @classmethod
    def build_parameters(
        cls,
        *,
        model_config: LinearLayerConfig | None,
        residual_dim: int | None,
        blend_initial_alpha: float,
        build_model: Callable[..., Module | None],
    ) -> _PairwiseResidualParameters:
        if not cls.USES_MIX_COEFFICIENT:
            return _PairwiseResidualParameters()
        if model_config is None:
            raw_mix_coefficient = cast(
                Tensor,
                cls.initial_raw_mix_coefficient(blend_initial_alpha),
            )
            return _PairwiseResidualParameters(
                raw_weight=nn.Parameter(raw_mix_coefficient),
            )

        coefficient_dim = cast(int, residual_dim)
        coefficient_model = cast(
            "LinearLayer",
            build_model(
                model_config,
                input_dim=coefficient_dim * 2,
                output_dim=coefficient_dim,
            ),
        )
        initial_raw_mix_coefficient = cast(
            Tensor,
            cls.initial_raw_mix_coefficient(blend_initial_alpha),
        ).item()
        nn.init.zeros_(coefficient_model.weight_params)
        bias_params = cast(Tensor, coefficient_model.bias_params)
        nn.init.constant_(
            bias_params,
            initial_raw_mix_coefficient,
        )
        return _PairwiseResidualParameters(model=coefficient_model)

    @staticmethod
    def _resolve_raw_mix_coefficient(
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        if connection.model is not None:
            coefficient_model_input = torch.cat((current, previous), dim=-1)
            return connection.model(coefficient_model_input)
        raw_mix_coefficient = connection.raw_weight
        connection.VALIDATOR.validate_raw_mix_coefficient(
            raw_mix_coefficient,
            connection.option,
        )
        return cast(Tensor, raw_mix_coefficient)

    @classmethod
    @abstractmethod
    def forward(
        cls,
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: object | None = None,
    ) -> Tensor:
        """Apply this residual option to the current and previous sources."""


class AdditiveResidual(PairwiseResidual):
    @classmethod
    def forward(
        cls,
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: object | None = None,
    ) -> Tensor:
        return current + previous


class WeightedResidual(PairwiseResidual):
    USES_MIX_COEFFICIENT = True

    @staticmethod
    def initial_raw_mix_coefficient(
        blend_initial_alpha: float,
    ) -> Tensor:
        return torch.tensor(0.0)

    @classmethod
    def forward(
        cls,
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: object | None = None,
    ) -> Tensor:
        raw_mix_coefficient = cls._resolve_raw_mix_coefficient(
            connection,
            current,
            previous,
        )
        residual_weight = torch.tanh(raw_mix_coefficient)
        return previous + residual_weight * current


class WeightedBlendResidual(PairwiseResidual):
    USES_MIX_COEFFICIENT = True
    DEFAULT_INITIAL_ALPHA = 0.9

    @staticmethod
    def initial_raw_mix_coefficient(
        blend_initial_alpha: float,
    ) -> Tensor:
        initial_logit = math.log(blend_initial_alpha / (1.0 - blend_initial_alpha))
        return torch.tensor(initial_logit)

    @classmethod
    def forward(
        cls,
        connection: ResidualConnection,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: object | None = None,
    ) -> Tensor:
        raw_mix_coefficient = cls._resolve_raw_mix_coefficient(
            connection,
            current,
            previous,
        )
        current_blend_coefficient = torch.sigmoid(raw_mix_coefficient)
        previous_blend_coefficient = 1.0 - current_blend_coefficient
        current_blend_contribution = current_blend_coefficient * current
        previous_blend_contribution = previous_blend_coefficient * previous
        return current_blend_contribution + previous_blend_contribution
