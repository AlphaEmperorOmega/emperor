from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from emperor.layers._state import LayerState
from emperor.memory import MemoryPositionOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import LayerConfig
    from emperor.memory import MemoryInterface


def _implements_memory_interface(model: object) -> TypeGuard[MemoryInterface]:
    return callable(model) and hasattr(model, "memory_position_option")


class LayerMemoryDelegate(Module):
    """Own dynamic-memory construction and position-aware application."""

    def __init__(
        self,
        layer_config: LayerConfig,
    ) -> None:
        super().__init__()
        self.config = layer_config.memory_config
        input_dim = layer_config.input_dim
        output_dim = layer_config.output_dim
        if input_dim is None or output_dim is None:
            raise ValueError(
                "Layer memory requires resolved input and output dimensions."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model: MemoryInterface | None = self.__build_model()

    def __build_model(self) -> MemoryInterface | None:
        model = self._build_from_config(
            self.config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        if model is None:
            return None
        if not _implements_memory_interface(model):
            raise TypeError(
                "memory_config must build a model implementing MemoryInterface."
            )
        return model

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
        if state.row_layout is not None:
            state.row_layout = state.row_layout.with_context_sharing_restricted()
        if self.model.memory_position_option != position:
            return state
        state.hidden = self.model(state.hidden)
        return state
