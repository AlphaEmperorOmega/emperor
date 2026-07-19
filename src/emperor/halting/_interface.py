"""Owner-facing interface implemented by halting strategies."""

from typing import Protocol, TypeVar, runtime_checkable

from torch import Tensor

StateT = TypeVar("StateT")


@runtime_checkable
class HaltingInterface(Protocol[StateT]):
    """StickBreaking-shaped lifecycle consumed by halting owners."""

    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]: ...

    def finalize_weighted_accumulation(
        self,
        state: StateT,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]: ...
