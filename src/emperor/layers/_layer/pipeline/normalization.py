from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from torch import nn

from emperor.layers._layer.pipeline.normalization_variants import (
    DynamicErf,
    DynamicISRU,
    DynamicTanh,
)
from emperor.layers._layer.validation import LayerNormalizationDelegateValidator
from emperor.layers._options import LayerNormPositionOptions, NormalizationOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import LayerConfig
    from emperor.layers._state import LayerState


class LayerNormalizationDelegate(Module):
    """Own configured normalization construction and position dispatch."""

    VALIDATOR = LayerNormalizationDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.__initialize_from_config()
        self.module = self.__build_normalization()

    def __initialize_from_config(self) -> None:
        self.position: LayerNormPositionOptions = self.cfg.layer_norm_position
        self.normalization: NormalizationOptions = (
            self.cfg.normalization or NormalizationOptions.RMS_NORM
        )
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.dimension = self.__resolve_dimension(self.input_dim, self.output_dim)

    def __build_normalization(self) -> nn.Module | None:
        if self.dimension is None:
            return None
        match self.normalization:
            case NormalizationOptions.RMS_NORM:
                return nn.RMSNorm(self.dimension, eps=1e-5)
            case NormalizationOptions.LAYER_NORM:
                return nn.LayerNorm(self.dimension, eps=1e-5)
            case NormalizationOptions.DYNAMIC_TANH:
                return DynamicTanh(self.dimension)
            case NormalizationOptions.DERF:
                return DynamicErf(self.dimension)
            case NormalizationOptions.DYISRU:
                return DynamicISRU(self.dimension)
            case _:
                assert_never(self.normalization)

    def __resolve_dimension(self, input_dim: int, output_dim: int) -> int | None:
        if self.position == LayerNormPositionOptions.DISABLED:
            return None
        if self.position == LayerNormPositionOptions.BEFORE:
            return input_dim
        return output_dim

    def before_model(self, state: LayerState) -> LayerState:
        return self.__normalize_at_position(state, LayerNormPositionOptions.BEFORE)

    def after_model(self, state: LayerState) -> LayerState:
        return self.__normalize_at_position(state, LayerNormPositionOptions.DEFAULT)

    def after_residual(self, state: LayerState) -> LayerState:
        return self.__normalize_at_position(state, LayerNormPositionOptions.AFTER)

    def __normalize_at_position(
        self,
        state: LayerState,
        position: LayerNormPositionOptions,
    ) -> LayerState:
        module = self.module
        if self.position != position or module is None:
            return state
        state.hidden = module(state.hidden)
        return state
