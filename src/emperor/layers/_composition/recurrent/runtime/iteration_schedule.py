from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from emperor.layers._composition.recurrent.config import (
    HierarchicalReasoningModelRecurrentConfig,
    RecurrentCompositionConfig,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.layers._composition.recurrent.validation.iteration_schedule import (
    RecurrentIterationScheduleValidator,
)


@dataclass(frozen=True)
class RecurrentIterationScheduleSnapshot:
    iteration_unit: str
    initial_iterations: int
    maximum_iterations: int
    active_iterations: int
    maximum_transition_count: int
    active_transition_count: int
    gradient_transition_count: int | None
    no_gradient_transition_count: int
    iteration_increment: int
    forward_calls_before_iteration_increment: int
    forward_call_progress: int
    complete: bool
    smooth_iteration_growth: bool
    settled_iterations: int
    transitioning: bool
    transition_source_iterations: int | None
    transition_target_iterations: int | None
    transition_forward_index: int | None
    transition_forward_count: int
    transition_weight: float


@dataclass(frozen=True)
class RecurrentBranchExecutionPlan:
    transition_count: int
    no_gradient_transition_count: int


@dataclass(frozen=True)
class RecurrentIterationExecutionPlan:
    common_prefix_transition_count: int
    target_branch: RecurrentBranchExecutionPlan
    transition_weight: float

    @property
    def transitioning(self) -> bool:
        return False


@dataclass(frozen=True)
class RecurrentSmoothHandoffExecutionPlan(RecurrentIterationExecutionPlan):
    source_branch: RecurrentBranchExecutionPlan

    @property
    def transitioning(self) -> bool:
        return True


@dataclass(frozen=True)
class _RecurrentIterationProfile:
    iteration_unit: str
    maximum_iterations: int
    transitions_per_iteration: int
    default_gradient_transition_count: int | None


class RecurrentIterationSchedule(nn.Module):
    """Own recurrent loop growth and its gradient-tracked suffix."""

    VALIDATOR = RecurrentIterationScheduleValidator

    def __init__(self, config: RecurrentCompositionConfig) -> None:
        super().__init__()
        self.VALIDATOR.validate_config(config)
        profile = self.__profile_from_config(config)

        self.iteration_unit = profile.iteration_unit
        self.maximum_iterations = profile.maximum_iterations
        self.transitions_per_iteration = profile.transitions_per_iteration
        self.maximum_transition_count = (
            self.maximum_iterations * self.transitions_per_iteration
        )
        self.initial_iterations = cast(int, config.initial_iterations)
        self.gradient_transition_count = config.gradient_transition_count
        self.iteration_increment = cast(int, config.iteration_increment)
        self.forward_calls_before_iteration_increment = cast(
            int,
            config.forward_calls_before_iteration_increment,
        )
        self.smooth_iteration_growth = config.smooth_iteration_growth_flag is True
        self.transition_forward_count = (
            self.forward_calls_before_iteration_increment // 2
            if self.smooth_iteration_growth
            else 0
        )
        self.__configured_no_gradient_transition_count = (
            config.no_gradient_transition_count
        )
        self.__default_gradient_transition_count = (
            profile.default_gradient_transition_count
        )
        self.__saturation_progress = self.__compute_saturation_progress()
        self.__forward_call_progress = 0

        self.active_iterations = self.initial_iterations
        self.settled_iterations = self.initial_iterations
        self.transitioning = False
        self.transition_source_iterations: int | None = None
        self.transition_target_iterations: int | None = None
        self.transition_forward_index: int | None = None
        self.transition_weight = 0.0
        self.active_transition_count = 0
        self.complete = False
        self.no_gradient_transition_count = 0
        self.__refresh_runtime(0)

        self.register_buffer(
            "forward_call_progress",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_load_state_dict_pre_hook(self.__prepare_checkpoint)
        self.register_load_state_dict_post_hook(self.__restore_runtime)

    @staticmethod
    def __profile_from_config(
        config: RecurrentCompositionConfig,
    ) -> _RecurrentIterationProfile:
        if isinstance(config, RecurrentLayerConfig):
            return _RecurrentIterationProfile(
                iteration_unit="transition",
                maximum_iterations=cast(int, config.max_steps),
                transitions_per_iteration=1,
                default_gradient_transition_count=None,
            )
        if isinstance(config, TinyRecursiveModelRecurrentConfig):
            transitions_per_iteration = (
                cast(int, config.latent_updates_per_answer_update) + 1
            )
            return _RecurrentIterationProfile(
                iteration_unit="answer_cycle",
                maximum_iterations=cast(int, config.answer_update_count),
                transitions_per_iteration=transitions_per_iteration,
                default_gradient_transition_count=transitions_per_iteration,
            )
        hierarchical_config = cast(
            HierarchicalReasoningModelRecurrentConfig,
            config,
        )
        return _RecurrentIterationProfile(
            iteration_unit="high_cycle",
            maximum_iterations=cast(int, hierarchical_config.high_cycles),
            transitions_per_iteration=cast(int, hierarchical_config.low_cycles) + 1,
            default_gradient_transition_count=2,
        )

    def __compute_saturation_progress(self) -> int:
        remaining_iterations = self.maximum_iterations - self.initial_iterations
        if remaining_iterations <= 0:
            return 0
        if self.smooth_iteration_growth:
            return (
                remaining_iterations * self.forward_calls_before_iteration_increment
                + self.transition_forward_count
            )
        remaining_iterations_with_rounding_offset = (
            remaining_iterations + self.iteration_increment - 1
        )
        required_increments = (
            remaining_iterations_with_rounding_offset // self.iteration_increment
        )
        return required_increments * self.forward_calls_before_iteration_increment

    def __active_iterations_for_progress(self, progress: int) -> int:
        completed_increments = progress // self.forward_calls_before_iteration_increment
        scheduled_iterations = (
            self.initial_iterations + completed_increments * self.iteration_increment
        )
        return min(self.maximum_iterations, scheduled_iterations)

    def __refresh_runtime(self, progress: int) -> None:
        if self.smooth_iteration_growth:
            self.__refresh_smooth_runtime(progress)
        else:
            self.active_iterations = self.__active_iterations_for_progress(progress)
            self.settled_iterations = self.active_iterations
            self.transitioning = False
            self.transition_source_iterations = None
            self.transition_target_iterations = None
            self.transition_forward_index = None
            self.transition_weight = 0.0
        self.active_transition_count = (
            self.active_iterations * self.transitions_per_iteration
        )
        self.complete = (
            not self.transitioning
            and self.settled_iterations == self.maximum_iterations
        )
        self.__refresh_gradient_window()

    def __refresh_smooth_runtime(self, progress: int) -> None:
        cadence = self.forward_calls_before_iteration_increment
        if progress < cadence:
            self.__set_stable_smooth_depth(self.initial_iterations)
            return

        growth_interval = progress // cadence
        interval_offset = progress % cadence
        source_depth = min(
            self.maximum_iterations,
            self.initial_iterations + growth_interval - 1,
        )
        if (
            source_depth < self.maximum_iterations
            and interval_offset < self.transition_forward_count
        ):
            transition_index = interval_offset + 1
            self.settled_iterations = source_depth
            self.active_iterations = source_depth + 1
            self.transitioning = True
            self.transition_source_iterations = source_depth
            self.transition_target_iterations = source_depth + 1
            self.transition_forward_index = transition_index
            self.transition_weight = transition_index / self.transition_forward_count
            return

        self.__set_stable_smooth_depth(min(self.maximum_iterations, source_depth + 1))

    def __set_stable_smooth_depth(self, depth: int) -> None:
        self.settled_iterations = depth
        self.active_iterations = depth
        self.transitioning = False
        self.transition_source_iterations = None
        self.transition_target_iterations = None
        self.transition_forward_index = None
        self.transition_weight = 0.0

    def __refresh_gradient_window(self) -> None:
        configured_no_gradient_count = self.__configured_no_gradient_transition_count
        if configured_no_gradient_count is not None:
            self.no_gradient_transition_count = configured_no_gradient_count
            return
        gradient_transition_count = self.gradient_transition_count
        if gradient_transition_count is None:
            gradient_transition_count = self.__default_gradient_transition_count
        if gradient_transition_count is None:
            self.no_gradient_transition_count = 0
            return
        self.no_gradient_transition_count = (
            self.active_transition_count - gradient_transition_count
        )

    def record_successful_forward(self) -> None:
        """Advance after one complete direct invocation of the recurrent owner."""
        if self.complete:
            return
        progress = min(
            self.__forward_call_progress + 1,
            self.__saturation_progress,
        )
        self.__forward_call_progress = progress
        self.forward_call_progress.fill_(progress)
        self.__refresh_runtime(progress)

    def gradient_context(
        self,
        transition_index: int,
    ) -> AbstractContextManager[None]:
        if not self.tracks_gradients(transition_index):
            return torch.no_grad()
        return nullcontext()

    def starts_gradient_suffix(self, transition_index: int) -> bool:
        return (
            self.no_gradient_transition_count > 0
            and transition_index == self.no_gradient_transition_count
        )

    def tracks_gradients(self, transition_index: int) -> bool:
        return transition_index >= self.no_gradient_transition_count

    def snapshot(self) -> RecurrentIterationScheduleSnapshot:
        return RecurrentIterationScheduleSnapshot(
            iteration_unit=self.iteration_unit,
            initial_iterations=self.initial_iterations,
            maximum_iterations=self.maximum_iterations,
            active_iterations=self.active_iterations,
            maximum_transition_count=self.maximum_transition_count,
            active_transition_count=self.active_transition_count,
            gradient_transition_count=self.gradient_transition_count,
            no_gradient_transition_count=self.no_gradient_transition_count,
            iteration_increment=self.iteration_increment,
            forward_calls_before_iteration_increment=(
                self.forward_calls_before_iteration_increment
            ),
            forward_call_progress=self.__forward_call_progress,
            complete=self.complete,
            smooth_iteration_growth=self.smooth_iteration_growth,
            settled_iterations=self.settled_iterations,
            transitioning=self.transitioning,
            transition_source_iterations=self.transition_source_iterations,
            transition_target_iterations=self.transition_target_iterations,
            transition_forward_index=self.transition_forward_index,
            transition_forward_count=self.transition_forward_count,
            transition_weight=self.transition_weight,
        )

    def execution_plan(self) -> RecurrentIterationExecutionPlan:
        if not self.transitioning:
            return RecurrentIterationExecutionPlan(
                common_prefix_transition_count=0,
                target_branch=RecurrentBranchExecutionPlan(
                    transition_count=self.active_transition_count,
                    no_gradient_transition_count=self.no_gradient_transition_count,
                ),
                transition_weight=0.0,
            )

        source_iterations = self.transition_source_iterations
        target_iterations = self.transition_target_iterations
        gradient_transition_count = self.gradient_transition_count
        if TYPE_CHECKING:
            # Smooth-growth validation and runtime refresh enforce these invariants.
            assert source_iterations is not None
            assert target_iterations is not None
            assert gradient_transition_count is not None
        source_transition_count = source_iterations * self.transitions_per_iteration
        target_transition_count = target_iterations * self.transitions_per_iteration
        source_no_gradient_count = source_transition_count - gradient_transition_count
        target_no_gradient_count = target_transition_count - gradient_transition_count
        execution_plan = RecurrentSmoothHandoffExecutionPlan(
            common_prefix_transition_count=source_no_gradient_count,
            source_branch=RecurrentBranchExecutionPlan(
                transition_count=source_transition_count,
                no_gradient_transition_count=source_no_gradient_count,
            ),
            target_branch=RecurrentBranchExecutionPlan(
                transition_count=target_transition_count,
                no_gradient_transition_count=target_no_gradient_count,
            ),
            transition_weight=self.transition_weight,
        )
        self.VALIDATOR.validate_smooth_handoff_source_branch(
            execution_plan.source_branch
        )
        return execution_plan

    def __prepare_checkpoint(
        self,
        _module: nn.Module,
        state_dict: dict[str, Tensor],
        prefix: str,
        _local_metadata: dict[str, object],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_messages: list[str],
    ) -> None:
        checkpoint_key = f"{prefix}forward_call_progress"
        incoming_progress = state_dict.get(checkpoint_key)
        if incoming_progress is None:
            state_dict[checkpoint_key] = self.forward_call_progress.new_zeros(())
            return
        progress = self.VALIDATOR.validate_checkpoint_progress(
            incoming_progress,
            checkpoint_key=checkpoint_key,
        )
        state_dict[checkpoint_key] = incoming_progress.new_tensor(
            min(progress, self.__saturation_progress)
        )

    def __restore_runtime(
        self,
        _module: nn.Module,
        _incompatible_keys: object,
    ) -> None:
        self.__forward_call_progress = int(self.forward_call_progress.item())
        self.__refresh_runtime(self.__forward_call_progress)
