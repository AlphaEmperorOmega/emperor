from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.config import RecurrentLayerConfig
from emperor.layers._composition.recurrent.validation import RecurrentLayerValidator
from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._row_layout import RowLayout


@dataclass(frozen=True)
class _RecurrentState:
    hidden: Tensor
    fixed_input: Tensor
    loss: Tensor | None
    context_state: LayerState
    row_layout: RowLayout | None = None
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False


class RecurrentLayer(RecurrentCompositionAbstract):
    VALIDATOR = RecurrentLayerValidator
    supports_recurrent_diagnostics = True

    def __init__(
        self,
        cfg: RecurrentLayerConfig,
        overrides: RecurrentLayerConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: RecurrentLayerConfig
        self.max_steps: int = self.cfg.max_steps
        self.reinject_original_hidden_flag: bool = (
            self.cfg.reinject_original_hidden_flag is True
        )
        self.block_config: ConfigBase | None = self.cfg.block_config
        self._initialize_transition_gradient_window(
            default_no_gradient_transition_count=0,
        )

        self.block_model = self._build_transition_model(self.block_config)

    @property
    def recurrent_diagnostic_step_limit(self) -> int:
        return self.max_steps

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)

        recurrent_state = self.__run_recurrent_steps(state)
        finalized_hidden, finalized_loss = self._finalize_recurrent_halting(
            recurrent_state.hidden,
            recurrent_state.loss,
            recurrent_state.halting_state,
        )
        state.hidden = finalized_hidden
        state.loss = finalized_loss
        return state

    def __run_recurrent_steps(
        self,
        layer_state: LayerState,
    ) -> _RecurrentState:
        recurrent_state = self.__initialize_recurrent_state(layer_state)
        for transition_index in range(self.max_steps):
            recurrent_state = self.__detach_evolving_state_at_gradient_boundary(
                recurrent_state, transition_index
            )
            with self._transition_gradient_context(transition_index):
                recurrent_state = self.__run_standard_transition(
                    recurrent_state, transition_index
                )

            if recurrent_state.all_items_halted:
                break

        return recurrent_state

    def __initialize_recurrent_state(
        self,
        layer_state: LayerState,
    ) -> _RecurrentState:
        return _RecurrentState(
            hidden=layer_state.hidden,
            fixed_input=layer_state.hidden,
            loss=layer_state.loss,
            context_state=layer_state,
            row_layout=self._recurrent_row_layout_for_transitions(layer_state),
        )

    def __detach_evolving_state_at_gradient_boundary(
        self,
        recurrent_state: _RecurrentState,
        transition_index: int,
    ) -> _RecurrentState:
        if not self._starts_gradient_suffix(transition_index):
            return recurrent_state
        detached_hidden = recurrent_state.hidden.detach()
        return replace(recurrent_state, hidden=detached_hidden)

    def __run_standard_transition(
        self,
        recurrent_state: _RecurrentState,
        transition_index: int,
    ) -> _RecurrentState:
        previous_hidden = recurrent_state.hidden
        transition_input = self.__maybe_reinject_original_hidden(
            previous_hidden, recurrent_state.fixed_input
        )
        halting_update_enabled = transition_index >= self.no_gradient_transition_count
        transition_result = self._run_recurrent_transition(
            recurrent_state,
            run_transition=self.block_model,
            transition_input=transition_input,
            previous_evolving_hidden=previous_hidden,
            loss=recurrent_state.loss,
            halting_update_enabled=halting_update_enabled,
        )
        return replace(
            recurrent_state,
            hidden=transition_result.hidden,
            loss=transition_result.loss,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __maybe_reinject_original_hidden(
        self,
        hidden: Tensor,
        fixed_input: Tensor,
    ) -> Tensor:
        if not self.reinject_original_hidden_flag:
            return hidden
        return hidden + fixed_input
