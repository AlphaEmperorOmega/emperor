from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn

from emperor.layers._composition.gate import LayerGate
from emperor.layers._options import LayerNormPositionOptions
from emperor.layers._support import LayerModuleBase, RowLayoutAwareModule
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from emperor.config import ConfigBase
    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.config import (
        RecurrentCompositionConfig,
    )
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState
    from emperor.nn import Module


class _RecurrentTransitionContext(Protocol):
    """Common schedule state consumed by the shared transition pipeline."""

    context_state: LayerState
    row_layout: RowLayout | None
    halting_state: HaltingStateBase | None


@dataclass(frozen=True)
class _RecurrentTransitionResult:
    hidden: Tensor
    loss: Tensor | None
    halting_state: HaltingStateBase | None
    all_items_halted: bool


class RecurrentCompositionAbstract(LayerModuleBase, ABC):
    """Private stable interface implemented by every recurrent variant."""

    supports_recurrent_diagnostics = False
    recurrent_diagnostic_step_limit = 1

    def __init__(
        self,
        cfg: RecurrentCompositionConfig,
        overrides: RecurrentCompositionConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: RecurrentCompositionConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.recurrent_layer_norm_position: LayerNormPositionOptions = (
            self.cfg.recurrent_layer_norm_position or LayerNormPositionOptions.DISABLED
        )
        self.gate_config = self.cfg.gate_config
        self.residual_config = self.cfg.residual_config
        self.halting_config = self.cfg.halting_config
        self.memory_config = self.cfg.memory_config
        self.recurrent_gate = self.__build_recurrent_gate()
        self.residual_connection = self.__build_residual_connection()
        self.halting_model = self.__build_halting_model()
        self.memory_model = self.__build_memory_model()
        self.recurrent_layer_norm_module = self.__build_recurrent_layer_norm()
        self._recurrent_diagnostic_observer: Callable[[Tensor, Tensor], None] | None = (
            None
        )

    def _build_transition_model(self, transition_config: ConfigBase) -> Module:
        return self._build_from_config(
            transition_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __build_recurrent_gate(self) -> Module | None:
        return self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )

    def __build_residual_connection(self) -> Module | None:
        return self._build_from_config(
            self.residual_config,
            residual_dim=self.output_dim,
        )

    def __build_halting_model(self) -> Module | None:
        return self._build_from_config(
            self.halting_config,
            input_dim=self.output_dim,
        )

    def __build_memory_model(self) -> Module | None:
        return self._build_from_config(
            self.memory_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __build_recurrent_layer_norm(self) -> nn.Module | None:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        return nn.LayerNorm(self.output_dim)

    def _initialize_transition_gradient_window(
        self,
        *,
        default_no_gradient_transition_count: int,
    ) -> None:
        configured_no_gradient_count = self.cfg.no_gradient_transition_count
        self.no_gradient_transition_count = (
            default_no_gradient_transition_count
            if configured_no_gradient_count is None
            else configured_no_gradient_count
        )

    def _transition_gradient_context(
        self,
        transition_index: int,
    ) -> AbstractContextManager[None]:
        if transition_index < self.no_gradient_transition_count:
            return torch.no_grad()
        return nullcontext()

    def _starts_gradient_suffix(self, transition_index: int) -> bool:
        return (
            self.no_gradient_transition_count > 0
            and transition_index == self.no_gradient_transition_count
        )

    def _set_recurrent_diagnostic_observer(
        self,
        observer: Callable[[Tensor, Tensor], None] | None,
    ) -> None:
        self._recurrent_diagnostic_observer = observer

    def _observe_recurrent_step(
        self,
        previous_hidden: Tensor,
        output_hidden: Tensor,
    ) -> None:
        observer = self._recurrent_diagnostic_observer
        if observer is not None:
            observer(previous_hidden, output_hidden)

    def _new_recurrent_initial_buffer(self, standard_deviation: float) -> Tensor:
        initial = torch.empty(self.output_dim)
        if standard_deviation == 0:
            return initial.zero_()
        return nn.init.trunc_normal_(
            initial,
            mean=0.0,
            std=standard_deviation,
            a=-2.0 * standard_deviation,
            b=2.0 * standard_deviation,
        )

    @staticmethod
    def _expand_recurrent_initial(initial: Tensor, hidden: Tensor) -> Tensor:
        view_shape = (1,) * (hidden.ndim - 1) + (hidden.shape[-1],)
        return initial.view(view_shape).expand_as(hidden)

    def _run_recurrent_transition(
        self,
        recurrent_state: _RecurrentTransitionContext,
        *,
        run_transition: Callable[[LayerState], LayerState],
        transition_input: Tensor,
        previous_evolving_hidden: Tensor,
        halting_update_enabled: bool,
        loss: Tensor | None = None,
    ) -> _RecurrentTransitionResult:
        transition_model_input = self.__maybe_apply_layer_norm_before(transition_input)
        transition_model_input = self.__maybe_apply_memory_before(
            transition_model_input
        )
        transition_state = replace(
            recurrent_state.context_state,
            hidden=transition_model_input,
            loss=loss,
            halting_state=None,
            row_layout=recurrent_state.row_layout,
        )
        output_state = run_transition(transition_state)
        self.VALIDATOR.validate_transition_output(
            output_state,
            transition_model_input,
            recurrent_state.row_layout,
            expected_feature_dim=self.output_dim,
        )
        candidate_hidden = self.__maybe_apply_memory_after(output_state.hidden)
        candidate_hidden = self.__maybe_apply_layer_norm_default(candidate_hidden)
        candidate_hidden = self.__maybe_apply_gate(
            candidate_hidden,
            recurrent_state.row_layout,
        )
        candidate_hidden = self.__maybe_apply_residual_connection(
            candidate_hidden,
            previous_evolving_hidden,
            recurrent_state.row_layout,
        )
        candidate_hidden = self.__maybe_apply_layer_norm_after(candidate_hidden)
        halting_state, output_hidden = self.__maybe_apply_halting(
            recurrent_state.halting_state,
            candidate_hidden,
            update_enabled=halting_update_enabled,
        )
        self._observe_recurrent_step(
            previous_evolving_hidden,
            output_hidden,
        )
        return _RecurrentTransitionResult(
            hidden=output_hidden,
            loss=output_state.loss,
            halting_state=halting_state,
            all_items_halted=self.__all_items_halted(halting_state),
        )

    def __maybe_apply_layer_norm_before(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_apply_memory_before(self, hidden: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.BEFORE_AFFINE,
        )

    def __maybe_apply_memory_after(self, hidden: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.AFTER_AFFINE,
        )

    def __maybe_apply_layer_norm_default(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_apply_gate(
        self,
        candidate_hidden: Tensor,
        row_layout: RowLayout | None,
    ) -> Tensor:
        if self.recurrent_gate is None:
            return candidate_hidden
        if isinstance(self.recurrent_gate, (LayerGate, RowLayoutAwareModule)):
            return self.recurrent_gate(candidate_hidden, row_layout=row_layout)
        return self.recurrent_gate(candidate_hidden)

    def __maybe_apply_residual_connection(
        self,
        candidate_hidden: Tensor,
        previous_hidden: Tensor,
        row_layout: RowLayout | None,
    ) -> Tensor:
        if self.residual_connection is None:
            return candidate_hidden
        return self.residual_connection(
            candidate_hidden,
            previous_hidden,
            row_layout=row_layout,
        )

    def __maybe_apply_layer_norm_after(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_apply_halting(
        self,
        previous_halting_state: HaltingStateBase | None,
        candidate_hidden: Tensor,
        *,
        update_enabled: bool,
    ) -> tuple[HaltingStateBase | None, Tensor]:
        if self.halting_model is None or not update_enabled:
            return previous_halting_state, candidate_hidden

        updated_halting_state, output_hidden = self.halting_model.update_halting_state(
            previous_halting_state,
            candidate_hidden,
        )
        self.VALIDATOR.validate_halting_output(output_hidden, candidate_hidden)
        return updated_halting_state, output_hidden

    def _finalize_recurrent_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.halting_model is None or halting_state is None:
            return hidden, loss

        finalized_hidden, halting_loss = (
            self.halting_model.finalize_weighted_accumulation(halting_state, hidden)
        )
        reduced_halting_loss = self._reduce_auxiliary_loss(halting_loss)
        return finalized_hidden, self._accumulate_auxiliary_loss(
            loss, reduced_halting_loss
        )

    def _recurrent_row_layout_for_transitions(
        self,
        state: LayerState,
    ) -> RowLayout | None:
        row_layout = state.row_layout
        if row_layout is None:
            return None
        if self.halting_model is not None or self.memory_model is not None:
            return row_layout.with_context_sharing_restricted()
        return row_layout

    @staticmethod
    def __all_items_halted(halting_state: HaltingStateBase | None) -> bool:
        if halting_state is None or halting_state.halt_mask is None:
            return False
        return bool(halting_state.halt_mask.all().item())

    def _accumulate_recurrent_losses(
        self,
        initial_loss: Tensor | None,
        auxiliary_losses: list[Tensor],
    ) -> Tensor | None:
        accumulated_loss = initial_loss
        for auxiliary_loss in auxiliary_losses:
            accumulated_loss = self._accumulate_auxiliary_loss(
                accumulated_loss,
                auxiliary_loss,
            )
        return accumulated_loss

    @abstractmethod
    def forward(self, state: LayerState) -> LayerState:
        """Apply one configured recurrent composition to a LayerState."""
