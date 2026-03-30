from torch import Tensor
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from emperor.base.utils import Module


StateT = TypeVar("StateT")


class HaltingBase(Module, Generic[StateT], ABC):
    @abstractmethod
    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]: ...
