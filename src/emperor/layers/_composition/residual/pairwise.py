from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

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


class WeightedPairwiseResidualAbstract(ResidualConnectionAbstract):
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
        self.__initialize_scalar_weight_or_coefficient_model()

    def __initialize_scalar_weight_or_coefficient_model(self) -> None:
        if self.model_config is None:
            initial_raw_coefficient = self._initial_raw_mix_coefficient()
            self.raw_weight = nn.Parameter(initial_raw_coefficient)
            return

        self.model = self._build_from_config(
            self.model_config,
            input_dim=self.residual_dim * 2,
            output_dim=self.residual_dim,
        )

    @staticmethod
    @abstractmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        """Return the raw scalar coefficient used at initialization."""

    @abstractmethod
    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
    ) -> Tensor:
        """Compose two sources using a learned coefficient."""

    def _resolve_raw_mix_coefficient(
        self,
        current: Tensor,
        previous: Tensor,
    ) -> Tensor:
        coefficient_model = self.model
        if coefficient_model is None:
            self.VALIDATOR.validate_raw_mix_coefficient(self.raw_weight)
            return self.raw_weight

        coefficient_model_input = torch.cat((current, previous), dim=-1)
        from emperor.layers import LayerStack, LayerState

        if isinstance(coefficient_model, LayerStack):
            coefficient_input_state = LayerState(
                hidden=coefficient_model_input,
            )
            coefficient_output_state = coefficient_model(coefficient_input_state)
            raw_mix_coefficient = coefficient_output_state.hidden
            return raw_mix_coefficient
        return coefficient_model(coefficient_model_input)
