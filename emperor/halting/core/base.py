from torch import Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from emperor.base.utils import Module


StateT = TypeVar("StateT")


@dataclass
class HaltingStateBase:
    halt_mask: Tensor | None = field(default=None, init=False)


class HaltingBase(Module, Generic[StateT], ABC):
    @abstractmethod
    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]: ...
