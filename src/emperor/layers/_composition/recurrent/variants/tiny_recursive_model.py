from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.runtime.execution import (
    PreparedRecurrentTransition,
    RecurrentExecution,
)
from emperor.layers._composition.recurrent.validation import (
    TinyRecursiveModelRecurrentValidator,
)

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult
    from emperor.layers._composition.recurrent.config import (
        TinyRecursiveModelRecurrentConfig,
    )
    from emperor.layers._state import LayerState
    from emperor.nn import Module


@dataclass(frozen=True)
class _TinyRecursiveModelState:
    fixed_input: Tensor
    answer: Tensor
    latent: Tensor
    initial_loss: Tensor | None
    auxiliary_losses: tuple[Tensor, ...]
    context_state: LayerState
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    def output_hidden(self) -> Tensor:
        return self.answer


class TinyRecursiveModelRecurrent(RecurrentCompositionAbstract):
    """Apply one shared transition using the Tiny Recursive Model schedule."""

    VALIDATOR = TinyRecursiveModelRecurrentValidator
    supports_recurrent_diagnostics = True

    if TYPE_CHECKING:
        answer_initial: Tensor
        latent_initial: Tensor

    def __init__(
        self,
        cfg: TinyRecursiveModelRecurrentConfig,
        overrides: TinyRecursiveModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: TinyRecursiveModelRecurrentConfig
        self.block_config: ConfigBase = self.cfg.block_config
        self.latent_updates_per_answer_update: int = (
            self.cfg.latent_updates_per_answer_update
        )
        self.answer_update_count: int = self.cfg.answer_update_count
        self.initialization_standard_deviation: float = (
            self.cfg.initialization_standard_deviation
        )

        self.__register_initial_buffer("answer_initial")
        self.__register_initial_buffer("latent_initial")
        self.block_model: Module = self._build_transition_model(self.block_config)
        self.__recurrent_execution: RecurrentExecution[_TinyRecursiveModelState] = (
            RecurrentExecution()
        )

    def __register_initial_buffer(self, buffer_name: str) -> None:
        initial_buffer = self._new_recurrent_initial_buffer(
            self.initialization_standard_deviation
        )
        self.register_buffer(buffer_name, initial_buffer, persistent=True)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        self.VALIDATOR.validate_initial_buffers(
            state.hidden,
            answer_initial=self.answer_initial,
            latent_initial=self.latent_initial,
            expected_feature_dim=self.output_dim,
        )

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
    ) -> _TinyRecursiveModelState:
        fixed_input = layer_state.hidden
        return _TinyRecursiveModelState(
            fixed_input=fixed_input,
            answer=self._expand_recurrent_initial(
                self.answer_initial,
                fixed_input,
            ),
            latent=self._expand_recurrent_initial(
                self.latent_initial,
                fixed_input,
            ),
            initial_loss=branch_base_loss,
            auxiliary_losses=(),
            context_state=layer_state,
            transition_index=0,
        )

    @staticmethod
    def _detach_recurrent_execution_state(
        recurrent_state: _TinyRecursiveModelState,
    ) -> _TinyRecursiveModelState:
        return replace(
            recurrent_state,
            answer=recurrent_state.answer.detach(),
            latent=recurrent_state.latent.detach(),
        )

    def _prepare_recurrent_transition(
        self,
        recurrent_state: _TinyRecursiveModelState,
        *,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition:
        if self.__updates_answer(recurrent_state):
            return PreparedRecurrentTransition(
                run_transition=self.block_model,
                transition_input=recurrent_state.answer + recurrent_state.latent,
                previous_evolving_hidden=recurrent_state.answer,
                halting_update_enabled=tracks_gradients,
            )
        return PreparedRecurrentTransition(
            run_transition=self.block_model,
            transition_input=(
                recurrent_state.latent
                + recurrent_state.answer
                + recurrent_state.fixed_input
            ),
            previous_evolving_hidden=recurrent_state.latent,
            halting_update_enabled=False,
        )

    def _apply_recurrent_transition_result(
        self,
        recurrent_state: _TinyRecursiveModelState,
        transition_result: RecurrentTransitionResult,
    ) -> _TinyRecursiveModelState:
        auxiliary_losses = self.__with_reduced_transition_loss(
            recurrent_state.auxiliary_losses,
            transition_result.loss,
        )
        state_updates = (
            {"answer": transition_result.hidden}
            if self.__updates_answer(recurrent_state)
            else {"latent": transition_result.hidden}
        )
        return replace(
            recurrent_state,
            **state_updates,
            auxiliary_losses=auxiliary_losses,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __with_reduced_transition_loss(
        self,
        auxiliary_losses: tuple[Tensor, ...],
        transition_loss: Tensor | None,
    ) -> tuple[Tensor, ...]:
        if transition_loss is None:
            return auxiliary_losses
        return (*auxiliary_losses, self._reduce_auxiliary_loss(transition_loss))

    @staticmethod
    def _fork_recurrent_handoff_state(
        recurrent_state: _TinyRecursiveModelState,
    ) -> _TinyRecursiveModelState:
        return replace(recurrent_state)

    def _recurrent_branch_loss(
        self,
        recurrent_state: _TinyRecursiveModelState,
    ) -> Tensor | None:
        return self._accumulate_recurrent_losses(
            recurrent_state.initial_loss,
            recurrent_state.auxiliary_losses,
        )

    def __updates_answer(
        self,
        recurrent_state: _TinyRecursiveModelState,
    ) -> bool:
        transitions_per_iteration = (
            self.recurrent_iteration_schedule.transitions_per_iteration
        )
        phase_index = recurrent_state.transition_index % transitions_per_iteration
        return phase_index >= self.latent_updates_per_answer_update
