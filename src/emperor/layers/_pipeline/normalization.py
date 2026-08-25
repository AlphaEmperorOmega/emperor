from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from emperor.layers._options import LayerNormPositionOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import LayerConfig
    from emperor.layers._state import LayerState


class LayerNormalizationDelegate(Module):
    """Own Layer normalization construction and position dispatch."""

    def __init__(
        self,
        layer_config: LayerConfig,
    ) -> None:
        super().__init__()
        position = layer_config.layer_norm_position
        input_dim = layer_config.input_dim
        output_dim = layer_config.output_dim
        if position is None:
            raise ValueError("Layer normalization requires a resolved position.")
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer normalization requires resolved input and output dimensions."
            )
        self.position = position
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dimension = self.__resolve_dimension(
            self.input_dim,
            self.output_dim,
        )
        self.module: nn.LayerNorm | None = (
            None if self.dimension is None else nn.LayerNorm(self.dimension)
        )

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
