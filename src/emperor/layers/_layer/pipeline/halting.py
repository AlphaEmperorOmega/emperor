from __future__ import annotations

from typing import TYPE_CHECKING

from emperor.layers._layer.validation import LayerHaltingDelegateValidator
from emperor.layers._state import LayerState
from emperor.nn import Module

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._config import LayerConfig


class LayerHaltingDelegate(Module):
    """Own halting construction and the Layer halting lifecycle."""

    VALIDATOR = LayerHaltingDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.__initialize_from_config()
        self.model: HaltingInterface[HaltingStateBase] | None = self.__build_model()
        self.is_terminal = False

    def __initialize_from_config(self) -> None:
        self.config = self.cfg.halting_config
        self.output_dim: int = self.cfg.output_dim

    def __build_model(self) -> HaltingInterface[HaltingStateBase] | None:
        model = self._build_from_config(
            self.config,
            input_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_halting_model_interface(model)

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
            halting_state, state.hidden
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
