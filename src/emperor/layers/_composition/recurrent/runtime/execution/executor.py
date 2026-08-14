# Shared Recurrent Execution Module. Recurrent variant classes implement its
# Adapter Interface while retaining variant-specific state and transition knowledge.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

import torch

from emperor.layers._composition.recurrent.runtime.execution.interface import (
    RecurrentExecutionAdapter,
    RecurrentExecutionResult,
    _StateT,
)
from emperor.layers._composition.recurrent.runtime.execution.runtime_state import (
    RecurrentRuntimeStateGuard,
)
from emperor.layers._composition.recurrent.runtime.iteration_schedule import (
    RecurrentBranchExecutionPlan,
    RecurrentIterationExecutionPlan,
    RecurrentIterationSchedule,
    RecurrentSmoothHandoffExecutionPlan,
)
from emperor.layers._composition.recurrent.validation import (
    RecurrentExecutionValidator,
)
from emperor.layers._state import LayerState

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.base import RecurrentTransitionResult


class RecurrentExecution(Generic[_StateT]):
    """Execute and commit recurrent schedules for any recurrent state Adapter."""

    VALIDATOR = RecurrentExecutionValidator
    __runtime_state_guard = RecurrentRuntimeStateGuard()

    def execute(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
    ) -> LayerState:
        self.VALIDATOR.validate_adapter_is_module(adapter)
        execution_plan = iteration_schedule.execution_plan()
        if isinstance(execution_plan, RecurrentSmoothHandoffExecutionPlan):
            return self.__execute_smooth_depth_handoff(
                adapter,
                layer_state,
                iteration_schedule,
                execution_plan,
            )
        execution_result = self.__execute_stable_plan(
            adapter,
            layer_state,
            execution_plan,
        )
        return self.__commit_execution_result(
            layer_state,
            iteration_schedule,
            execution_result,
        )

    @staticmethod
    def __commit_execution_result(
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
        execution_result: RecurrentExecutionResult,
    ) -> LayerState:
        layer_state.hidden = execution_result.hidden
        layer_state.loss = execution_result.loss
        iteration_schedule.record_successful_forward()
        return layer_state

    def __execute_stable_plan(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        execution_plan: RecurrentIterationExecutionPlan,
    ) -> RecurrentExecutionResult:
        recurrent_state = adapter._initialize_recurrent_execution_state(
            layer_state,
            branch_base_loss=layer_state.loss,
        )
        recurrent_state = self.__run_branch_suffix(
            adapter,
            recurrent_state,
            execution_plan.target_branch,
            observe_transitions=True,
        )
        return self.__finalize_branch(adapter, recurrent_state)

    def __execute_smooth_depth_handoff(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        iteration_schedule: RecurrentIterationSchedule,
        execution_plan: RecurrentSmoothHandoffExecutionPlan,
    ) -> LayerState:
        original_hidden = layer_state.hidden
        original_loss = layer_state.loss
        try:
            with self.__runtime_state_guard.rollback_handoff_on_failure(adapter):
                execution_result = self.__execute_committed_smooth_depth_handoff(
                    adapter,
                    layer_state,
                    execution_plan,
                )
                return self.__commit_execution_result(
                    layer_state,
                    iteration_schedule,
                    execution_result,
                )
        except BaseException:
            layer_state.hidden = original_hidden
            layer_state.loss = original_loss
            raise

    def __execute_committed_smooth_depth_handoff(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        layer_state: LayerState,
        execution_plan: RecurrentSmoothHandoffExecutionPlan,
    ) -> RecurrentExecutionResult:
        common_state = adapter._initialize_recurrent_execution_state(
            layer_state,
            branch_base_loss=None,
        )
        common_transition_count = execution_plan.common_prefix_transition_count
        common_state = self.__run_branch_suffix(
            adapter,
            common_state,
            RecurrentBranchExecutionPlan(
                transition_count=common_transition_count,
                no_gradient_transition_count=common_transition_count,
            ),
            observe_transitions=True,
        )

        source_result, target_state = (
            self.__advance_handoff_branches_through_shared_boundary(
                adapter,
                common_state,
                execution_plan.source_branch,
            )
        )

        target_state = self.__run_branch_suffix(
            adapter,
            target_state,
            execution_plan.target_branch,
            observe_transitions=True,
        )
        target_result = self.__finalize_branch(adapter, target_state)
        transition_weight = execution_plan.transition_weight
        blended_hidden = self.__blend_recurrent_branch_hidden(
            source_result.hidden,
            target_result.hidden,
            transition_weight,
        )
        blended_recurrent_loss = adapter._blend_recurrent_branch_losses(
            adapter._recurrent_branch_loss(common_state),
            source_result.loss,
            target_result.loss,
            transition_weight,
        )
        if blended_recurrent_loss is None:
            return RecurrentExecutionResult(blended_hidden, layer_state.loss)
        return RecurrentExecutionResult(
            blended_hidden,
            adapter._accumulate_auxiliary_loss(
                layer_state.loss,
                blended_recurrent_loss,
            ),
        )

    @staticmethod
    def __blend_recurrent_branch_hidden(
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        transition_weight: float,
    ) -> torch.Tensor:
        if transition_weight == 0.0:
            return source_hidden
        if transition_weight == 1.0:
            return target_hidden
        return (
            1.0 - transition_weight
        ) * source_hidden + transition_weight * target_hidden

    def __advance_handoff_branches_through_shared_boundary(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        common_state: _StateT,
        source_plan: RecurrentBranchExecutionPlan,
    ) -> tuple[RecurrentExecutionResult, _StateT]:
        source_boundary_state = self.__detach_state_at_gradient_boundary(
            adapter,
            common_state,
            source_plan,
        )
        transition_index = source_boundary_state.transition_index
        source_tracks_gradients = source_plan.tracks_gradients(transition_index)
        with source_plan.gradient_context(transition_index):
            prepared_transition = adapter._prepare_recurrent_transition(
                source_boundary_state,
                tracks_gradients=source_tracks_gradients,
            )

            def run_and_finalize_provisional_source_branch(
                source_transition_result: RecurrentTransitionResult,
            ) -> RecurrentExecutionResult:
                source_state = adapter._fork_recurrent_handoff_state(
                    source_boundary_state
                )
                source_state = adapter._apply_recurrent_transition_result(
                    source_state,
                    source_transition_result,
                )
                source_state = self.__run_branch_suffix(
                    adapter,
                    source_state,
                    source_plan,
                    observe_transitions=False,
                )
                return self.__finalize_branch(adapter, source_state)

            source_result, target_transition_result = (
                adapter._run_shared_handoff_boundary_transition(
                    source_boundary_state,
                    run_transition=prepared_transition.run_transition,
                    transition_input=prepared_transition.transition_input,
                    previous_evolving_hidden=(
                        prepared_transition.previous_evolving_hidden
                    ),
                    source_halting_update_enabled=(
                        prepared_transition.halting_update_enabled
                    ),
                    run_provisional_source_branch=(
                        run_and_finalize_provisional_source_branch
                    ),
                    loss=prepared_transition.loss,
                    residual_state=prepared_transition.residual_state,
                    residual_schedule=prepared_transition.residual_schedule,
                    transition_index=transition_index,
                )
            )

        target_state = adapter._fork_recurrent_handoff_state(source_boundary_state)
        target_state = adapter._apply_recurrent_transition_result(
            target_state,
            target_transition_result,
        )
        return source_result, target_state

    def __run_branch_suffix(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
        *,
        observe_transitions: bool,
    ) -> _StateT:
        while (
            recurrent_state.transition_index < branch_plan.transition_count
            and not recurrent_state.all_items_halted
        ):
            recurrent_state = self.__run_planned_transition(
                adapter,
                recurrent_state,
                branch_plan,
                observe_transition=observe_transitions,
            )
        return recurrent_state

    def __run_planned_transition(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
        *,
        observe_transition: bool,
    ) -> _StateT:
        recurrent_state = self.__detach_state_at_gradient_boundary(
            adapter,
            recurrent_state,
            branch_plan,
        )
        transition_index = recurrent_state.transition_index
        tracks_gradients = branch_plan.tracks_gradients(transition_index)
        with branch_plan.gradient_context(transition_index):
            prepared_transition = adapter._prepare_recurrent_transition(
                recurrent_state,
                tracks_gradients=tracks_gradients,
            )
            transition_result = adapter._run_recurrent_transition(
                recurrent_state,
                run_transition=prepared_transition.run_transition,
                transition_input=prepared_transition.transition_input,
                previous_evolving_hidden=(prepared_transition.previous_evolving_hidden),
                halting_update_enabled=(prepared_transition.halting_update_enabled),
                loss=prepared_transition.loss,
                residual_state=prepared_transition.residual_state,
                residual_schedule=prepared_transition.residual_schedule,
                transition_index=transition_index,
                observe_transition=observe_transition,
            )
            return adapter._apply_recurrent_transition_result(
                recurrent_state,
                transition_result,
            )

    def __detach_state_at_gradient_boundary(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
        branch_plan: RecurrentBranchExecutionPlan,
    ) -> _StateT:
        if not branch_plan.starts_gradient_suffix(recurrent_state.transition_index):
            return recurrent_state
        return adapter._detach_recurrent_execution_state(recurrent_state)

    def __finalize_branch(
        self,
        adapter: RecurrentExecutionAdapter[_StateT],
        recurrent_state: _StateT,
    ) -> RecurrentExecutionResult:
        finalized_hidden, finalized_loss = adapter._finalize_recurrent_halting(
            recurrent_state.output_hidden,
            adapter._recurrent_branch_loss(recurrent_state),
            recurrent_state.halting_state,
        )
        return RecurrentExecutionResult(finalized_hidden, finalized_loss)
