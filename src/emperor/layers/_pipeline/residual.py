from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.residual.base import ResidualConnectionAbstract
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._config import LayerConfig
    from emperor.layers._state import LayerState


class LayerResidualDelegate(Module):
    """Own residual construction, state creation, and Layer application."""

    def __init__(
        self,
        layer_config: LayerConfig,
    ) -> None:
        super().__init__()
        self.config = layer_config.residual_config
        output_dim = layer_config.output_dim
        if output_dim is None:
            raise ValueError(
                "Layer residual processing requires a resolved output_dim."
            )
        self.output_dim = output_dim
        self.connection: ResidualConnectionAbstract | None = (
            self.__build_residual_connection()
        )

    def __build_residual_connection(self) -> ResidualConnectionAbstract | None:
        residual_connection = self._build_from_config(
            self.config,
            residual_dim=self.output_dim,
        )
        if residual_connection is None:
            return None
        if not isinstance(residual_connection, ResidualConnectionAbstract):
            raise TypeError("residual_config must build a ResidualConnectionAbstract.")
        return residual_connection

    def new_state(self, initial_source: Tensor) -> ResidualState | None:
        if self.connection is None:
            return None
        return self.connection.new_state(initial_source)

    def apply_residual(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        if self.connection is None:
            return state
        state.hidden = self.connection(
            state.hidden,
            previous,
            residual_state=state.residual_state,
            row_layout=state.row_layout,
        )
        return state
