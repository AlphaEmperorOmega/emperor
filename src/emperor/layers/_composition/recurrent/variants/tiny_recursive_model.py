from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.validation import (
    TinyRecursiveModelRecurrentValidator,
)

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.config import (
        TinyRecursiveModelRecurrentConfig,
    )
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState
    from emperor.nn import Module


@dataclass(frozen=True)
class _TinyRecursiveModelState:
    fixed_input: Tensor
    answer: Tensor
    latent: Tensor
    context_state: LayerState
    row_layout: RowLayout | None
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False


class TinyRecursiveModelRecurrent(RecurrentCompositionAbstract):
    """Apply one shared transition using the Tiny Recursive Model schedule."""

    VALIDATOR = TinyRecursiveModelRecurrentValidator
    supports_recurrent_diagnostics = True

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
        transitions_per_answer_update = self.latent_updates_per_answer_update + 1
        self._initialize_transition_gradient_window(
            default_no_gradient_transition_count=(
                (self.answer_update_count - 1) * transitions_per_answer_update
            ),
        )
        self.initialization_standard_deviation: float = (
            self.cfg.initialization_standard_deviation
        )
        self.__register_initial_buffer("answer_initial")
        self.__register_initial_buffer("latent_initial")
        self.block_model: Module = self._build_transition_model(self.block_config)

    def __register_initial_buffer(self, buffer_name: str) -> None:
        initial_buffer = self._new_recurrent_initial_buffer(
            self.initialization_standard_deviation
        )
        self.register_buffer(buffer_name, initial_buffer, persistent=True)

    @property
    def recurrent_diagnostic_step_limit(self) -> int:
        return self.answer_update_count * (self.latent_updates_per_answer_update + 1)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        fixed_input = state.hidden
        self.VALIDATOR.validate_initial_buffers(
            fixed_input,
            answer_initial=self.answer_initial,
            latent_initial=self.latent_initial,
            expected_feature_dim=self.output_dim,
        )
        tiny_recursive_state = _TinyRecursiveModelState(
            fixed_input=fixed_input,
            answer=self._expand_recurrent_initial(self.answer_initial, fixed_input),
            latent=self._expand_recurrent_initial(self.latent_initial, fixed_input),
            context_state=state,
            row_layout=self._recurrent_row_layout_for_transitions(state),
            transition_index=0,
        )
        auxiliary_losses: list[Tensor] = []

        for _ in range(self.answer_update_count):
            tiny_recursive_state = self.__run_answer_cycle(
                tiny_recursive_state, auxiliary_losses
            )
            if tiny_recursive_state.all_items_halted:
                break

        accumulated_loss = self._accumulate_recurrent_losses(
            state.loss, auxiliary_losses
        )
        finalized_answer, finalized_loss = self._finalize_recurrent_halting(
            tiny_recursive_state.answer,
            accumulated_loss,
            tiny_recursive_state.halting_state,
        )
        state.hidden = finalized_answer
        state.loss = finalized_loss
        return state

    def __run_answer_cycle(
        self,
        tiny_recursive_state: _TinyRecursiveModelState,
        auxiliary_losses: list[Tensor],
    ) -> _TinyRecursiveModelState:
        for _ in range(self.latent_updates_per_answer_update):
            tiny_recursive_state = self.__run_latent_update(
                tiny_recursive_state,
                auxiliary_losses,
            )
        return self.__run_answer_update(tiny_recursive_state, auxiliary_losses)

    def __run_latent_update(
        self,
        tiny_recursive_state: _TinyRecursiveModelState,
        auxiliary_losses: list[Tensor],
    ) -> _TinyRecursiveModelState:
        transition_index = tiny_recursive_state.transition_index
        answer, previous_latent = self.__detach_evolving_state_at_gradient_boundary(
            transition_index,
            answer=tiny_recursive_state.answer,
            latent=tiny_recursive_state.latent,
        )
        with self._transition_gradient_context(transition_index):
            latent_transition_input = (
                previous_latent + answer + tiny_recursive_state.fixed_input
            )
            transition_result = self._run_recurrent_transition(
                tiny_recursive_state,
                run_transition=self.block_model,
                transition_input=latent_transition_input,
                previous_evolving_hidden=previous_latent,
                halting_update_enabled=False,
            )
            if transition_result.loss is not None:
                auxiliary_losses.append(
                    self._reduce_auxiliary_loss(transition_result.loss)
                )
        return replace(
            tiny_recursive_state,
            answer=answer,
            latent=transition_result.hidden,
            transition_index=transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __run_answer_update(
        self,
        tiny_recursive_state: _TinyRecursiveModelState,
        auxiliary_losses: list[Tensor],
    ) -> _TinyRecursiveModelState:
        transition_index = tiny_recursive_state.transition_index
        previous_answer, latent = self.__detach_evolving_state_at_gradient_boundary(
            transition_index,
            answer=tiny_recursive_state.answer,
            latent=tiny_recursive_state.latent,
        )
        with self._transition_gradient_context(transition_index):
            answer_transition_input = previous_answer + latent
            halting_update_enabled = (
                transition_index >= self.no_gradient_transition_count
            )
            transition_result = self._run_recurrent_transition(
                tiny_recursive_state,
                run_transition=self.block_model,
                transition_input=answer_transition_input,
                previous_evolving_hidden=previous_answer,
                halting_update_enabled=halting_update_enabled,
            )
            if transition_result.loss is not None:
                auxiliary_losses.append(
                    self._reduce_auxiliary_loss(transition_result.loss)
                )
        return replace(
            tiny_recursive_state,
            answer=transition_result.hidden,
            latent=latent,
            transition_index=transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __detach_evolving_state_at_gradient_boundary(
        self,
        transition_index: int,
        *,
        answer: Tensor,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self._starts_gradient_suffix(transition_index):
            return answer, latent
        return answer.detach(), latent.detach()
