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
        self.__configured_no_gradient_transition_count = (
            config.no_gradient_transition_count
        )
        self.__default_gradient_transition_count = (
            profile.default_gradient_transition_count
        )
        self.__saturation_progress = self.__compute_saturation_progress()
        self.__forward_call_progress = 0

        self.active_iterations = self.__active_iterations_for_progress(0)
        self.active_transition_count = (
            self.active_iterations * self.transitions_per_iteration
        )
        self.complete = self.active_iterations == self.maximum_iterations
        self.no_gradient_transition_count = 0
        self.__refresh_gradient_window()

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
        self.active_iterations = self.__active_iterations_for_progress(progress)
        self.active_transition_count = (
            self.active_iterations * self.transitions_per_iteration
        )
        self.complete = self.active_iterations == self.maximum_iterations
        self.__refresh_gradient_window()

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

    def __restore_runtime(
        self,
        _module: nn.Module,
        _incompatible_keys: object,
    ) -> None:
        self.__forward_call_progress = int(self.forward_call_progress.item())
        self.__refresh_runtime(self.__forward_call_progress)
