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
    HierarchicalReasoningModelRecurrentValidator,
)

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult
    from emperor.layers._composition.recurrent.config import (
        HierarchicalReasoningModelRecurrentConfig,
    )
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState
    from emperor.nn import Module


@dataclass(frozen=True)
class _HierarchicalReasoningModelState:
    fixed_input: Tensor
    high: Tensor
    low: Tensor
    initial_loss: Tensor | None
    auxiliary_losses: list[Tensor]
    context_state: LayerState
    row_layout: RowLayout | None
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False

    @property
    def output_hidden(self) -> Tensor:
        return self.high


class HierarchicalReasoningModelRecurrent(RecurrentCompositionAbstract):
    """Apply distinct high- and low-level transitions on nested clocks."""

    VALIDATOR = HierarchicalReasoningModelRecurrentValidator
    supports_recurrent_diagnostics = True

    if TYPE_CHECKING:
        high_initial: Tensor
        low_initial: Tensor

    def __init__(
        self,
        cfg: HierarchicalReasoningModelRecurrentConfig,
        overrides: HierarchicalReasoningModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        resolved_config = self.cfg
        if TYPE_CHECKING:
            # The validator has already enforced every required leaf field.
            assert isinstance(
                resolved_config,
                HierarchicalReasoningModelRecurrentConfig,
            )
            assert resolved_config.high_block_config is not None
            assert resolved_config.low_block_config is not None
            assert resolved_config.high_cycles is not None
            assert resolved_config.low_cycles is not None
            assert resolved_config.initialization_standard_deviation is not None
        self.high_block_config: ConfigBase = resolved_config.high_block_config
        self.low_block_config: ConfigBase = resolved_config.low_block_config
        self.high_cycles: int = resolved_config.high_cycles
        self.low_cycles: int = resolved_config.low_cycles
        self.initialization_standard_deviation: float = (
            resolved_config.initialization_standard_deviation
        )
        self.__register_initial_buffer("high_initial")
        self.__register_initial_buffer("low_initial")
        self.high_model: Module = self._build_transition_model(self.high_block_config)
        self.low_model: Module = self._build_transition_model(self.low_block_config)
        self.__recurrent_execution: RecurrentExecution[
            _HierarchicalReasoningModelState
        ] = RecurrentExecution()

    def __register_initial_buffer(self, buffer_name: str) -> None:
        initial_buffer = self._new_recurrent_initial_buffer(
            self.initialization_standard_deviation
        )
        self.register_buffer(buffer_name, initial_buffer, persistent=True)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        self.VALIDATOR.validate_initial_buffers(
            state.hidden,
            high_initial=self.high_initial,
            low_initial=self.low_initial,
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
    ) -> _HierarchicalReasoningModelState:
        fixed_input = layer_state.hidden
        return _HierarchicalReasoningModelState(
            fixed_input=fixed_input,
            high=self._expand_recurrent_initial(self.high_initial, fixed_input),
            low=self._expand_recurrent_initial(self.low_initial, fixed_input),
            initial_loss=branch_base_loss,
            auxiliary_losses=[],
            context_state=layer_state,
            row_layout=self._recurrent_row_layout_for_transitions(layer_state),
            transition_index=0,
        )

    @staticmethod
    def _detach_recurrent_execution_state(
        recurrent_state: _HierarchicalReasoningModelState,
    ) -> _HierarchicalReasoningModelState:
        return replace(
            recurrent_state,
            high=recurrent_state.high.detach(),
            low=recurrent_state.low.detach(),
        )

    def _prepare_recurrent_transition(
        self,
        recurrent_state: _HierarchicalReasoningModelState,
        *,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition:
        if self.__updates_high(recurrent_state):
            return PreparedRecurrentTransition(
                run_transition=self.high_model,
                transition_input=recurrent_state.high + recurrent_state.low,
                previous_evolving_hidden=recurrent_state.high,
                halting_update_enabled=tracks_gradients,
            )
        return PreparedRecurrentTransition(
            run_transition=self.low_model,
            transition_input=(
                recurrent_state.low + recurrent_state.high + recurrent_state.fixed_input
            ),
            previous_evolving_hidden=recurrent_state.low,
            halting_update_enabled=False,
        )

    def _apply_recurrent_transition_result(
        self,
        recurrent_state: _HierarchicalReasoningModelState,
        transition_result: RecurrentTransitionResult,
    ) -> _HierarchicalReasoningModelState:
        self.__append_reduced_transition_loss(
            recurrent_state.auxiliary_losses,
            transition_result.loss,
        )
        state_updates = (
            {"high": transition_result.hidden}
            if self.__updates_high(recurrent_state)
            else {"low": transition_result.hidden}
        )
        return replace(
            recurrent_state,
            **state_updates,
            transition_index=recurrent_state.transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __append_reduced_transition_loss(
        self,
        auxiliary_losses: list[Tensor],
        transition_loss: Tensor | None,
    ) -> None:
        if transition_loss is None:
            return
        auxiliary_losses.append(self._reduce_auxiliary_loss(transition_loss))

    @staticmethod
    def _fork_recurrent_handoff_state(
        recurrent_state: _HierarchicalReasoningModelState,
    ) -> _HierarchicalReasoningModelState:
        return replace(
            recurrent_state,
            auxiliary_losses=list(recurrent_state.auxiliary_losses),
        )

    def _recurrent_branch_loss(
        self,
        recurrent_state: _HierarchicalReasoningModelState,
    ) -> Tensor | None:
        return self._accumulate_recurrent_losses(
            recurrent_state.initial_loss,
            recurrent_state.auxiliary_losses,
        )

    def __updates_high(
        self,
        recurrent_state: _HierarchicalReasoningModelState,
    ) -> bool:
        transitions_per_iteration = (
            self.recurrent_iteration_schedule.transitions_per_iteration
        )
        phase_index = recurrent_state.transition_index % transitions_per_iteration
        return phase_index >= self.low_cycles
