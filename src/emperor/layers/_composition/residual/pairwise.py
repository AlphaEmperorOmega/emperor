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
from emperor.layers._support import RowLayoutAwareModule

if TYPE_CHECKING:
    from emperor.layers import LayerStack, LayerStackConfig
    from emperor.layers._row_layout import RowLayout
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
        initial_raw_coefficient = self._initial_raw_mix_coefficient()
        if self.model_config is None:
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
        affine_output = self.__coefficient_affine_output()
        nn.init.zeros_(affine_output.weight_params)
        bias_params = cast(Tensor, affine_output.bias_params)
        nn.init.constant_(bias_params, initial_raw_coefficient.item())

    def __coefficient_affine_output(self) -> LinearAbstract:
        from emperor.layers import LayerStack
        from emperor.linears import LinearAbstract

        coefficient_model = self.model
        if isinstance(coefficient_model, LinearAbstract):
            return coefficient_model
        if isinstance(coefficient_model, LayerStack):
            output_model = coefficient_model[-1].model
            if isinstance(output_model, LinearAbstract):
                return output_model
        raise TypeError(
            "weighted residual coefficient model must end in LinearAbstract, got "
            f"{type(coefficient_model).__name__}."
        )

    @staticmethod
    @abstractmethod
    def _initial_raw_mix_coefficient() -> Tensor:
        """Return the raw scalar coefficient used at initialization."""

    def _resolve_raw_mix_coefficient(
        self,
        current: Tensor,
        previous: Tensor,
        row_layout: RowLayout | None,
    ) -> Tensor:
        coefficient_model = self.model
        if coefficient_model is not None:
            coefficient_model_input = torch.cat((current, previous), dim=-1)
            from emperor.layers import LayerStack, LayerState

            if isinstance(coefficient_model, LayerStack):
                coefficient_state = LayerState(
                    hidden=coefficient_model_input,
                    row_layout=row_layout,
                )
                return coefficient_model(coefficient_state).hidden
            if isinstance(coefficient_model, RowLayoutAwareModule):
                return coefficient_model(
                    coefficient_model_input,
                    row_layout=row_layout,
                )
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
        row_layout: RowLayout | None = None,
    ) -> Tensor:
        """Compose two sources using a learned coefficient."""
