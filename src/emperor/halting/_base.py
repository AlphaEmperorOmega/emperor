from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from torch import Tensor

from emperor.halting._validation import StickBreakingValidator
from emperor.nn import Module

StateT = TypeVar("StateT")


@dataclass
class HaltingStateBase:
    halt_mask: Tensor | None = field(default=None, init=False)


class HaltingBase(Module, Generic[StateT], ABC):
    VALIDATOR = StickBreakingValidator

    @abstractmethod
    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]: ...
