from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from torch import Tensor

from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualState,
)
from emperor.layers._composition.residual.config import (
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)

if TYPE_CHECKING:
    from emperor.layers import LayerStack, LayerStackConfig
    from emperor.linears import LinearAbstract, LinearLayerConfig


class PairwiseResidualAbstract(ResidualConnectionAbstract):
    """Residual Implementation that supports pairwise diagnostics."""

    supports_pairwise_diagnostics = True


class WeightedPairwiseResidualAbstract(PairwiseResidualAbstract):
    """Shared learned-coefficient mechanics for weighted pairwise variants."""

    def __init__(
        self,
        cfg: WeightedResidualConfig | WeightedBlendResidualConfig,
        overrides: WeightedResidualConfig | WeightedBlendResidualConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.model_config: LayerStackConfig | LinearLayerConfig | None = (
            self.cfg.model_config
        )
        self.raw_weight: nn.Parameter | None = None
        self.model: LayerStack | LinearAbstract | None = None
        self.__initialize_coefficient()

    def __initialize_coefficient(self) -> None:
        if self.model_config is None:
            initial_raw_coefficient = self._initial_raw_mix_coefficient()
            self.raw_weight = nn.Parameter(initial_raw_coefficient)
            return

        coefficient_dim = cast(int, self.residual_dim)
        self.model = cast(
            "LayerStack | LinearAbstract",
            self._build_from_config(
                self.model_config,
                input_dim=coefficient_dim * 2,
                output_dim=coefficient_dim,
            ),
        )

    @staticmethod
    @abstractmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        """Return the raw scalar coefficient used at initialization."""

    def _resolve_raw_mix_coefficient(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        coefficient_model = self.model
        if coefficient_model is not None:
            coefficient_model_input = torch.cat((current, previous), dim=-1)
            from emperor.layers import LayerStack, LayerState

            if isinstance(coefficient_model, LayerStack):
                coefficient_state = LayerState(
                    hidden=coefficient_model_input,
                )
                return coefficient_model(coefficient_state).hidden
            return coefficient_model(coefficient_model_input)
        self.VALIDATOR.validate_raw_mix_coefficient(self.raw_weight)
        return cast(Tensor, self.raw_weight)

    @abstractmethod
    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        """Compose two sources using a learned coefficient."""
