from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from emperor.layers._layer.validation import LayerNormalizationDelegateValidator
from emperor.layers._options import LayerNormPositionOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import LayerConfig
    from emperor.layers._state import LayerState


class LayerNormalizationDelegate(Module):
    """Own Layer normalization construction and position dispatch."""

    VALIDATOR = LayerNormalizationDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.__initialize_from_config()
        self.module = self.__build_layer_norm()

    def __initialize_from_config(self) -> None:
        self.position: LayerNormPositionOptions = self.cfg.layer_norm_position
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.dimension = self.__resolve_dimension(self.input_dim, self.output_dim)

    def __build_layer_norm(self) -> nn.LayerNorm | None:
        if self.dimension is None:
            return None
        return nn.LayerNorm(self.dimension)

    def __resolve_dimension(self, input_dim: int, output_dim: int) -> int | None:
        if self.position == LayerNormPositionOptions.DISABLED:
            return None
        if self.position == LayerNormPositionOptions.BEFORE:
            return input_dim
        return output_dim

    def before_model(self, state: LayerState) -> LayerState:
        module = self.module
        if self.position != LayerNormPositionOptions.BEFORE or module is None:
            return state
        state.hidden = module(state.hidden)
        return state

    def after_model(self, state: LayerState) -> LayerState:
        module = self.module
        if self.position != LayerNormPositionOptions.DEFAULT or module is None:
            return state
        state.hidden = module(state.hidden)
        return state

    def after_residual(self, state: LayerState) -> LayerState:
        module = self.module
        if self.position != LayerNormPositionOptions.AFTER or module is None:
            return state
        state.hidden = module(state.hidden)
        return state
