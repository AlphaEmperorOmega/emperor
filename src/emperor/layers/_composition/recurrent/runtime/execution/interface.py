"""Shared Interface and records for generalized Recurrent Execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from torch import Tensor

from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult
    from emperor.layers._composition.recurrent.runtime.residual_schedule import (
        RecurrentResidualSchedule,
    )
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._row_layout import RowLayout


@dataclass(frozen=True)
class RecurrentExecutionResult:
    hidden: Tensor
    loss: Tensor | None


class RecurrentExecutionState(Protocol):
    """State facts required by the generalized Recurrent Execution Interface."""

    @property
    def context_state(self) -> LayerState: ...

    @property
    def row_layout(self) -> RowLayout | None: ...

    @property
    def transition_index(self) -> int: ...

    @property
    def halting_state(self) -> HaltingStateBase | None: ...

    @property
    def all_items_halted(self) -> bool: ...

    @property
    def output_hidden(self) -> Tensor: ...


@dataclass(frozen=True)
class PreparedRecurrentTransition:
    """Variant-prepared request for the shared recurrent transition pipeline."""

    run_transition: Callable[[LayerState], LayerState]
    transition_input: Tensor
    previous_evolving_hidden: Tensor
    halting_update_enabled: bool
    loss: Tensor | None = None
    residual_state: ResidualState | None = None
    residual_schedule: RecurrentResidualSchedule | None = None


_StateT = TypeVar("_StateT", bound=RecurrentExecutionState)
_ProvisionalSourceBranchOutput = TypeVar("_ProvisionalSourceBranchOutput")


class RecurrentExecutionAdapter(Protocol[_StateT]):
    """Recurrent variant ``nn.Module`` Interface consumed by Recurrent Execution."""

    def _run_recurrent_transition(
        self,
        recurrent_state: _StateT,
        *,
        run_transition: Callable[[LayerState], LayerState],
        transition_input: Tensor,
        previous_evolving_hidden: Tensor,
        halting_update_enabled: bool,
        loss: Tensor | None = None,
        residual_state: ResidualState | None = None,
        residual_schedule: RecurrentResidualSchedule | None = None,
        transition_index: int = 0,
        observe_transition: bool = True,
    ) -> RecurrentTransitionResult: ...

    def _run_shared_handoff_boundary_transition(
        self,
        recurrent_state: _StateT,
        *,
        run_transition: Callable[[LayerState], LayerState],
        transition_input: Tensor,
        previous_evolving_hidden: Tensor,
        source_halting_update_enabled: bool,
        run_provisional_source_branch: Callable[
            [RecurrentTransitionResult],
            _ProvisionalSourceBranchOutput,
        ],
        loss: Tensor | None = None,
        residual_state: ResidualState | None = None,
        residual_schedule: RecurrentResidualSchedule | None = None,
        transition_index: int = 0,
    ) -> tuple[_ProvisionalSourceBranchOutput, RecurrentTransitionResult]: ...

    def _finalize_recurrent_halting(
        self,
        hidden: Tensor,
        loss: Tensor | None,
        halting_state: HaltingStateBase | None,
    ) -> tuple[Tensor, Tensor | None]: ...

    def _halting_usage_tracking_context(
        self,
        *,
        enabled: bool,
    ) -> AbstractContextManager[None]: ...

    def _blend_recurrent_branch_losses(
        self,
        common_loss: Tensor | None,
        source_loss: Tensor | None,
        target_loss: Tensor | None,
        transition_weight: float,
    ) -> Tensor | None: ...

    def _accumulate_auxiliary_loss(
        self,
        loss: Tensor | None,
        auxiliary_loss: Tensor,
    ) -> Tensor: ...

    def _initialize_recurrent_execution_state(
        self,
        layer_state: LayerState,
        *,
        branch_base_loss: Tensor | None,
    ) -> _StateT: ...

    def _detach_recurrent_execution_state(
        self,
        recurrent_state: _StateT,
    ) -> _StateT: ...

    def _prepare_recurrent_transition(
        self,
        recurrent_state: _StateT,
        *,
        tracks_gradients: bool,
    ) -> PreparedRecurrentTransition: ...

    def _apply_recurrent_transition_result(
        self,
        recurrent_state: _StateT,
        transition_result: RecurrentTransitionResult,
    ) -> _StateT: ...

    def _fork_recurrent_handoff_state(
        self,
        recurrent_state: _StateT,
    ) -> _StateT: ...

    def _recurrent_branch_loss(
        self,
        recurrent_state: _StateT,
    ) -> Tensor | None: ...
