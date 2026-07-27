from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._composition.recurrent.base import RecurrentCompositionAbstract
from emperor.layers._composition.recurrent.validation import (
    HierarchicalReasoningModelRecurrentValidator,
)

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
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
    context_state: LayerState
    row_layout: RowLayout | None
    transition_index: int
    halting_state: HaltingStateBase | None = None
    all_items_halted: bool = False


class HierarchicalReasoningModelRecurrent(RecurrentCompositionAbstract):
    """Apply distinct high- and low-level transitions on nested clocks."""

    VALIDATOR = HierarchicalReasoningModelRecurrentValidator
    supports_recurrent_diagnostics = True

    def __init__(
        self,
        cfg: HierarchicalReasoningModelRecurrentConfig,
        overrides: HierarchicalReasoningModelRecurrentConfig | None = None,
    ) -> None:
        super().__init__(cfg, overrides)
        self.cfg: HierarchicalReasoningModelRecurrentConfig
        self.high_block_config: ConfigBase = self.cfg.high_block_config
        self.low_block_config: ConfigBase = self.cfg.low_block_config
        self.high_cycles: int = self.cfg.high_cycles
        self.low_cycles: int = self.cfg.low_cycles
        total_transition_count = self.high_cycles * (self.low_cycles + 1)
        self._initialize_transition_gradient_window(
            default_no_gradient_transition_count=total_transition_count - 2,
        )
        self.initialization_standard_deviation: float = (
            self.cfg.initialization_standard_deviation
        )
        self.__register_initial_buffer("high_initial")
        self.__register_initial_buffer("low_initial")
        self.high_model: Module = self._build_transition_model(self.high_block_config)
        self.low_model: Module = self._build_transition_model(self.low_block_config)

    def __register_initial_buffer(self, buffer_name: str) -> None:
        initial_buffer = self._new_recurrent_initial_buffer(
            self.initialization_standard_deviation
        )
        self.register_buffer(buffer_name, initial_buffer, persistent=True)

    @property
    def recurrent_diagnostic_step_limit(self) -> int:
        return self.high_cycles * (self.low_cycles + 1)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)
        fixed_input = state.hidden
        self.VALIDATOR.validate_initial_buffers(
            fixed_input,
            high_initial=self.high_initial,
            low_initial=self.low_initial,
            expected_feature_dim=self.output_dim,
        )
        hierarchical_state = self.__initialize_recurrent_state(state, fixed_input)
        auxiliary_losses: list[Tensor] = []

        for _ in range(self.high_cycles):
            hierarchical_state = self.__run_high_cycle(
                hierarchical_state, auxiliary_losses
            )
            if hierarchical_state.all_items_halted:
                break

        accumulated_loss = self._accumulate_recurrent_losses(
            state.loss, auxiliary_losses
        )
        finalized_high, finalized_loss = self._finalize_recurrent_halting(
            hierarchical_state.high,
            accumulated_loss,
            hierarchical_state.halting_state,
        )
        state.hidden = finalized_high
        state.loss = finalized_loss
        return state

    def __initialize_recurrent_state(
        self,
        layer_state: LayerState,
        fixed_input: Tensor,
    ) -> _HierarchicalReasoningModelState:
        return _HierarchicalReasoningModelState(
            fixed_input=fixed_input,
            high=self._expand_recurrent_initial(self.high_initial, fixed_input),
            low=self._expand_recurrent_initial(self.low_initial, fixed_input),
            context_state=layer_state,
            row_layout=self._recurrent_row_layout_for_transitions(layer_state),
            transition_index=0,
        )

    def __run_high_cycle(
        self,
        hierarchical_state: _HierarchicalReasoningModelState,
        auxiliary_losses: list[Tensor],
    ) -> _HierarchicalReasoningModelState:
        for _ in range(self.low_cycles):
            hierarchical_state = self.__run_low_transition(
                hierarchical_state,
                auxiliary_losses,
            )
        return self.__run_high_transition(hierarchical_state, auxiliary_losses)

    def __run_low_transition(
        self,
        hierarchical_state: _HierarchicalReasoningModelState,
        auxiliary_losses: list[Tensor],
    ) -> _HierarchicalReasoningModelState:
        hierarchical_state = self.__detach_evolving_state_at_gradient_boundary(
            hierarchical_state
        )
        transition_index = hierarchical_state.transition_index
        with self._transition_gradient_context(transition_index):
            previous_low = hierarchical_state.low
            low_transition_input = (
                previous_low + hierarchical_state.high + hierarchical_state.fixed_input
            )
            transition_result = self._run_recurrent_transition(
                hierarchical_state,
                run_transition=self.low_model,
                transition_input=low_transition_input,
                previous_evolving_hidden=previous_low,
                halting_update_enabled=False,
            )
            if transition_result.loss is not None:
                auxiliary_losses.append(
                    self._reduce_auxiliary_loss(transition_result.loss)
                )
        return replace(
            hierarchical_state,
            low=transition_result.hidden,
            transition_index=transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __run_high_transition(
        self,
        hierarchical_state: _HierarchicalReasoningModelState,
        auxiliary_losses: list[Tensor],
    ) -> _HierarchicalReasoningModelState:
        hierarchical_state = self.__detach_evolving_state_at_gradient_boundary(
            hierarchical_state
        )
        transition_index = hierarchical_state.transition_index
        with self._transition_gradient_context(transition_index):
            previous_high = hierarchical_state.high
            high_transition_input = previous_high + hierarchical_state.low
            halting_update_enabled = (
                transition_index >= self.no_gradient_transition_count
            )
            transition_result = self._run_recurrent_transition(
                hierarchical_state,
                run_transition=self.high_model,
                transition_input=high_transition_input,
                previous_evolving_hidden=previous_high,
                halting_update_enabled=halting_update_enabled,
            )
            if transition_result.loss is not None:
                auxiliary_losses.append(
                    self._reduce_auxiliary_loss(transition_result.loss)
                )
        return replace(
            hierarchical_state,
            high=transition_result.hidden,
            transition_index=transition_index + 1,
            halting_state=transition_result.halting_state,
            all_items_halted=transition_result.all_items_halted,
        )

    def __detach_evolving_state_at_gradient_boundary(
        self,
        hierarchical_state: _HierarchicalReasoningModelState,
    ) -> _HierarchicalReasoningModelState:
        if not self._starts_gradient_suffix(hierarchical_state.transition_index):
            return hierarchical_state
        return replace(
            hierarchical_state,
            high=hierarchical_state.high.detach(),
            low=hierarchical_state.low.detach(),
        )
