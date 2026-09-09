from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.config import RecurrentLayerConfig
from emperor.layers._composition.recurrent.runtime.execution import (
    PreparedRecurrentTransition,
    RecurrentExecution,
)
from emperor.layers._composition.recurrent.validation import RecurrentLayerValidator
from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult
    from emperor.layers._composition.residual.base import ResidualState


@dataclass(frozen=True)
class _StandardRecurrentState:
    hidden: Tensor
    fixed_input: Tensor
    loss: Tensor | None
    context_state: LayerState
    transition_index: int
    residual_state: ResidualState | None = None
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    def output_hidden(self) -> Tensor:
        return self.hidden


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
        self.block_config: ConfigBase = self.cfg.block_config

        self.recurrent_residual_schedule = self._build_recurrent_residual_schedule(
            self.max_steps
        )
        self.block_model = self._build_transition_model(self.block_config)
        self.__recurrent_execution: RecurrentExecution[_StandardRecurrentState] = (
            RecurrentExecution()
        )

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        return self.__recurrent_execution.execute(
            self,
            state,
            self.recurrent_iteration_schedule,
        )

    def _initialize_recurrent_execution_state(
        self,
        layer_state: LayerState,
        *,
        branch_base_loss: Tensor | None,
    ) -> _StandardRecurrentState:
        return _StandardRecurrentState(
            hidden=layer_state.hidden,
            fixed_input=layer_state.hidden,
            loss=branch_base_loss,
            context_state=layer_state,
            transition_index=0,
            residual_state=self.__initialize_recurrent_residual_state(layer_state),
        )

    def __initialize_recurrent_residual_state(
        self,
        layer_state: LayerState,
    ) -> ResidualState | None:
        residual_schedule = self.recurrent_residual_schedule
        residual_connection = self.residual_connection
        if residual_schedule is None or residual_connection is None:
            return None
        return residual_schedule.new_state(
            residual_connection,
            layer_state.hidden,
        )

    @staticmethod
    def _detach_recurrent_execution_state(
        recurrent_state: _StandardRecurrentState,
    ) -> _StandardRecurrentState:
        return replace(
            recurrent_state,
            hidden=recurrent_state.hidden.detach(),
        )

    def _prepare_recurrent_transition(
        self,
        recurrent_state: _StandardRecurrentState,
        *,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition:
        previous_hidden = recurrent_state.hidden
        transition_input = self.__maybe_reinject_original_hidden(
            previous_hidden,
            recurrent_state.fixed_input,
        )
        return PreparedRecurrentTransition(
            run_transition=self.block_model,
            transition_input=transition_input,
            previous_evolving_hidden=previous_hidden,
            halting_update_enabled=tracks_gradients,
            loss=recurrent_state.loss,
            residual_state=recurrent_state.residual_state,
            residual_schedule=self.recurrent_residual_schedule,
        )

    def __maybe_reinject_original_hidden(
        self,
        hidden: Tensor,
        fixed_input: Tensor,
    ) -> Tensor:
        if not self.reinject_original_hidden_flag:
            return hidden
        return hidden + fixed_input

    @staticmethod
    def _apply_recurrent_transition_result(
        recurrent_state: _StandardRecurrentState,
        transition_result: RecurrentTransitionResult,
    ) -> _StandardRecurrentState:
        return replace(
            recurrent_state,
            hidden=transition_result.hidden,
            loss=transition_result.loss,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def _fork_recurrent_handoff_state(
        self,
        recurrent_state: _StandardRecurrentState,
    ) -> _StandardRecurrentState:
        residual_schedule = self.recurrent_residual_schedule
        residual_state = recurrent_state.residual_state
        if residual_schedule is None:
            return replace(recurrent_state)
        return replace(
            recurrent_state,
            residual_state=residual_schedule.fork_state(residual_state),
        )

    @staticmethod
    def _recurrent_branch_loss(
        recurrent_state: _StandardRecurrentState,
    ) -> Tensor | None:
        return recurrent_state.loss
