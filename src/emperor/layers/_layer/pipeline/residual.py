from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._layer.validation import LayerResidualDelegateValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import ResidualConnectionAbstract
    from emperor.layers._config import LayerConfig
    from emperor.layers._state import LayerState


class LayerResidualDelegate(Module):
    """Own residual construction, contract validation, and Layer application."""

    VALIDATOR = LayerResidualDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.config = self.cfg.residual_config
        self.output_dim: int = self.cfg.output_dim

        self.connection: ResidualConnectionAbstract | None = (
            self.__build_residual_connection()
        )

    def __build_residual_connection(self) -> ResidualConnectionAbstract | None:
        residual_connection = self._build_from_config(
            self.config,
            residual_dim=self.output_dim,
        )
        connection = self.VALIDATOR.validate_built_residual_connection_type(
            residual_connection
        )
        if connection is None:
            return None
        self.VALIDATOR.validate_forward_local_state_lifecycle_requirement(connection)
        return connection

    def apply_residual(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        connection = self.connection
        if connection is None:
            return state
        return connection.apply_to_layer_state(state, previous)
