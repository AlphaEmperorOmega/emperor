from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from emperor.layers._state import LayerState
from emperor.nn import Module

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._config import LayerConfig
    from emperor.layers._row_layout import RowLayout


def _implements_halting_interface(
    model: object,
) -> TypeGuard[HaltingInterface[HaltingStateBase]]:
    return callable(getattr(model, "update_halting_state", None)) and callable(
        getattr(model, "finalize_weighted_accumulation", None)
    )


class LayerHaltingDelegate(Module):
    """Own halting construction and the Layer halting lifecycle."""

    def __init__(
        self,
        layer_config: LayerConfig,
    ) -> None:
        super().__init__()
        self.config = layer_config.halting_config
        output_dim = layer_config.output_dim
        if output_dim is None:
            raise ValueError("Layer halting requires a resolved output_dim.")
        self.output_dim = output_dim
        self.model: HaltingInterface[HaltingStateBase] | None = self.__build_model()
        self.is_terminal = False

    def __build_model(self) -> HaltingInterface[HaltingStateBase] | None:
        model = self._build_from_config(
            self.config,
            input_dim=self.output_dim,
        )
        if model is None:
            return None
        if not _implements_halting_interface(model):
            raise TypeError(
                "halting_config must build a model implementing HaltingInterface."
            )
        return model

    def bind_shared(self, model: HaltingInterface[HaltingStateBase]) -> None:
        self.model = model

    def mark_as_terminal_layer(self) -> None:
        self.is_terminal = True

    def should_skip(self, state: LayerState) -> bool:
        if self.model is None or state.halting_state is None:
            return False
        return self.__is_complete(state.halting_state)

    def apply_halting(self, state: LayerState) -> LayerState:
        model = self.model
        if model is None:
            return state

        halting_state, halting_output = model.update_halting_state(
            state.halting_state, state.hidden
        )
        state.halting_state = halting_state
        if self.is_terminal or self.__is_complete(halting_state):
            return self.__finalize(state, model, halting_state)
        state.hidden = halting_output
        return state

    def restrict_row_layout(
        self,
        row_layout: RowLayout | None,
    ) -> RowLayout | None:
        if row_layout is None or self.model is None:
            return row_layout
        return row_layout.with_context_sharing_restricted()

    @staticmethod
    def __is_complete(halting_state: HaltingStateBase | None) -> bool:
        if halting_state is None or halting_state.halt_mask is None:
            return False
        return bool(halting_state.halt_mask.all().item())

    def __finalize(
        self,
        state: LayerState,
        model: HaltingInterface[HaltingStateBase],
        halting_state: HaltingStateBase,
    ) -> LayerState:
        state.hidden, halting_loss = model.finalize_weighted_accumulation(
            halting_state,
            state.hidden,
        )
        auxiliary_loss = self.__maybe_reduce_halting_loss(halting_loss)
        state.loss = self.__maybe_accumulate_auxiliary_loss(state.loss, auxiliary_loss)
        return state

    @staticmethod
    def __maybe_reduce_halting_loss(halting_loss: Tensor) -> Tensor:
        if halting_loss.dim() == 0:
            return halting_loss
        return halting_loss.mean()

    @staticmethod
    def __maybe_accumulate_auxiliary_loss(
        accumulated_loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor:
        if accumulated_loss is None:
            return auxiliary_loss
        return accumulated_loss + auxiliary_loss
