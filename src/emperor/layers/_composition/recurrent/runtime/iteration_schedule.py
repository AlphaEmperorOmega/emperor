from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast

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

    def gradient_context(self, transition_index: int) -> AbstractContextManager[None]:
        if self.tracks_gradients(transition_index):
            return nullcontext()
        return torch.no_grad()

    def starts_gradient_suffix(self, transition_index: int) -> bool:
        return (
            self.no_gradient_transition_count > 0
            and transition_index == self.no_gradient_transition_count
        )

    def tracks_gradients(self, transition_index: int) -> bool:
        return transition_index >= self.no_gradient_transition_count


@dataclass(frozen=True)
class RecurrentIterationExecutionPlan:
    target_branch: RecurrentBranchExecutionPlan


@dataclass(frozen=True)
class RecurrentSmoothHandoffExecutionPlan(RecurrentIterationExecutionPlan):
    common_prefix_transition_count: int
    source_branch: RecurrentBranchExecutionPlan
    transition_weight: float


@dataclass(frozen=True)
class RecurrentNestedSmoothHandoffExecutionPlan(RecurrentIterationExecutionPlan):
    source_branch: RecurrentBranchExecutionPlan
    transition_weight: float


@dataclass(frozen=True)
class _RecurrentIterationProfile:
    iteration_unit: str
    maximum_iterations: int
    transitions_per_iteration: int
    default_gradient_transition_count: int | None


@dataclass(frozen=True)
class _RecurrentIterationRuntime:
    settled_iterations: int
    active_iterations: int
    active_transition_count: int
    no_gradient_transition_count: int
    complete: bool


@dataclass(frozen=True)
class _RecurrentSmoothHandoffRuntime(_RecurrentIterationRuntime):
    source_iterations: int
    target_iterations: int
    transition_forward_index: int
    transition_weight: float


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
        self.__full_gradient_smooth_handoff = (
            self.smooth_iteration_growth
            and self.__configured_no_gradient_transition_count == 0
            and self.gradient_transition_count is None
        )
        self.__default_gradient_transition_count = (
            profile.default_gradient_transition_count
        )
        self.__saturation_progress = self.__compute_saturation_progress()

        self.register_buffer(
            "forward_call_progress",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_load_state_dict_pre_hook(self.__prepare_checkpoint)

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

    def __runtime_for_progress(self, progress: int) -> _RecurrentIterationRuntime:
        if self.smooth_iteration_growth:
            return self.__smooth_runtime_for_progress(progress)
        return self.__stable_runtime(
            self.__active_iterations_for_progress(progress),
        )

    def __smooth_runtime_for_progress(
        self,
        progress: int,
    ) -> _RecurrentIterationRuntime:
        cadence = self.forward_calls_before_iteration_increment
        if progress < cadence:
            return self.__stable_runtime(self.initial_iterations)

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
            target_depth = source_depth + 1
            active_transition_count = target_depth * self.transitions_per_iteration
            return _RecurrentSmoothHandoffRuntime(
                settled_iterations=source_depth,
                active_iterations=target_depth,
                active_transition_count=active_transition_count,
                no_gradient_transition_count=(
                    self.__no_gradient_transition_count(active_transition_count)
                ),
                complete=False,
                source_iterations=source_depth,
                target_iterations=target_depth,
                transition_forward_index=transition_index,
                transition_weight=(transition_index / self.transition_forward_count),
            )

        return self.__stable_runtime(
            min(self.maximum_iterations, source_depth + 1),
        )

    def __stable_runtime(self, depth: int) -> _RecurrentIterationRuntime:
        active_transition_count = depth * self.transitions_per_iteration
        return _RecurrentIterationRuntime(
            settled_iterations=depth,
            active_iterations=depth,
            active_transition_count=active_transition_count,
            no_gradient_transition_count=(
                self.__no_gradient_transition_count(active_transition_count)
            ),
            complete=depth == self.maximum_iterations,
        )

    def __no_gradient_transition_count(self, active_transition_count: int) -> int:
        configured_no_gradient_count = self.__configured_no_gradient_transition_count
        if configured_no_gradient_count is not None:
            return configured_no_gradient_count
        gradient_transition_count = self.gradient_transition_count
        if gradient_transition_count is None:
            gradient_transition_count = self.__default_gradient_transition_count
        if gradient_transition_count is None:
            return 0
        return active_transition_count - gradient_transition_count

    def __current_runtime(self) -> _RecurrentIterationRuntime:
        return self.__runtime_for_progress(self.__current_progress())

    def __current_progress(self) -> int:
        return int(self.forward_call_progress.item())

    @property
    def active_iterations(self) -> int:
        return self.__current_runtime().active_iterations

    @property
    def active_transition_count(self) -> int:
        return self.__current_runtime().active_transition_count

    @property
    def no_gradient_transition_count(self) -> int:
        return self.__current_runtime().no_gradient_transition_count

    @property
    def complete(self) -> bool:
        return self.__current_runtime().complete

    def record_successful_forward(self) -> None:
        """Advance after one complete direct invocation of the recurrent owner."""
        runtime = self.__current_runtime()
        if runtime.complete:
            return
        progress = min(
            self.__current_progress() + 1,
            self.__saturation_progress,
        )
        self.forward_call_progress.fill_(progress)

    def snapshot(self) -> RecurrentIterationScheduleSnapshot:
        progress = self.__current_progress()
        runtime = self.__runtime_for_progress(progress)
        if isinstance(runtime, _RecurrentSmoothHandoffRuntime):
            transition_source_iterations = runtime.source_iterations
            transition_target_iterations = runtime.target_iterations
            transition_forward_index = runtime.transition_forward_index
            transition_weight = runtime.transition_weight
        else:
            transition_source_iterations = None
            transition_target_iterations = None
            transition_forward_index = None
            transition_weight = 0.0
        return RecurrentIterationScheduleSnapshot(
            iteration_unit=self.iteration_unit,
            initial_iterations=self.initial_iterations,
            maximum_iterations=self.maximum_iterations,
            active_iterations=runtime.active_iterations,
            maximum_transition_count=self.maximum_transition_count,
            active_transition_count=runtime.active_transition_count,
            gradient_transition_count=self.gradient_transition_count,
            no_gradient_transition_count=runtime.no_gradient_transition_count,
            iteration_increment=self.iteration_increment,
            forward_calls_before_iteration_increment=(
                self.forward_calls_before_iteration_increment
            ),
            forward_call_progress=progress,
            complete=runtime.complete,
            smooth_iteration_growth=self.smooth_iteration_growth,
            settled_iterations=runtime.settled_iterations,
            transitioning=isinstance(runtime, _RecurrentSmoothHandoffRuntime),
            transition_source_iterations=transition_source_iterations,
            transition_target_iterations=transition_target_iterations,
            transition_forward_index=transition_forward_index,
            transition_forward_count=self.transition_forward_count,
            transition_weight=transition_weight,
        )

    def execution_plan(self) -> RecurrentIterationExecutionPlan:
        runtime = self.__current_runtime()
        if not isinstance(runtime, _RecurrentSmoothHandoffRuntime):
            return RecurrentIterationExecutionPlan(
                target_branch=RecurrentBranchExecutionPlan(
                    transition_count=runtime.active_transition_count,
                    no_gradient_transition_count=(runtime.no_gradient_transition_count),
                ),
            )

        source_transition_count = (
            runtime.source_iterations * self.transitions_per_iteration
        )
        source_no_gradient_count = self.__no_gradient_transition_count(
            source_transition_count
        )
        if self.__full_gradient_smooth_handoff:
            return RecurrentNestedSmoothHandoffExecutionPlan(
                source_branch=RecurrentBranchExecutionPlan(
                    transition_count=source_transition_count,
                    no_gradient_transition_count=source_no_gradient_count,
                ),
                target_branch=RecurrentBranchExecutionPlan(
                    transition_count=runtime.active_transition_count,
                    no_gradient_transition_count=runtime.no_gradient_transition_count,
                ),
                transition_weight=runtime.transition_weight,
            )
        return RecurrentSmoothHandoffExecutionPlan(
            common_prefix_transition_count=source_no_gradient_count,
            source_branch=RecurrentBranchExecutionPlan(
                transition_count=source_transition_count,
                no_gradient_transition_count=source_no_gradient_count,
            ),
            target_branch=RecurrentBranchExecutionPlan(
                transition_count=runtime.active_transition_count,
                no_gradient_transition_count=runtime.no_gradient_transition_count,
            ),
            transition_weight=runtime.transition_weight,
        )

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
