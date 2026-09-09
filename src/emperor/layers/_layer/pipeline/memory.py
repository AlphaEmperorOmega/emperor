from __future__ import annotations

from typing import TYPE_CHECKING

from emperor.layers._layer.validation import LayerMemoryDelegateValidator
from emperor.layers._state import LayerState
from emperor.memory import MemoryPositionOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import LayerConfig
    from emperor.memory import MemoryInterface


class LayerMemoryDelegate(Module):
    """Own dynamic-memory construction and position-aware application."""

    VALIDATOR = LayerMemoryDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.config = self.cfg.memory_config
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim

        self.model: MemoryInterface | None = self.__build_model()

    def __build_model(self) -> MemoryInterface | None:
        model = self._build_from_config(
            self.config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_memory_model_interface(model)

    def bind_shared(self, model: MemoryInterface) -> None:
        self.model = model

    def before_model(self, state: LayerState) -> LayerState:
        return self.__apply_at_position(state, MemoryPositionOptions.BEFORE_AFFINE)

    def after_model(self, state: LayerState) -> LayerState:
        return self.__apply_at_position(state, MemoryPositionOptions.AFTER_AFFINE)

    def __apply_at_position(
        self,
        state: LayerState,
        position: MemoryPositionOptions,
    ) -> LayerState:
        if self.model is None:
            return state
        if self.model.memory_position_option != position:
            return state
        state.hidden = self.model(state.hidden)
        return state
