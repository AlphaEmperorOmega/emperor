from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from torch import Tensor

from emperor.halting._interface import HaltingInterface
from emperor.nn import Module

StateT = TypeVar("StateT")


@dataclass
class HaltingStateBase:
    halt_mask: Tensor | None = field(default=None, init=False)
    output_hidden: Tensor = field(init=False, repr=False)
    accumulated_hidden: Tensor = field(init=False, repr=False)
    continuation_probability: Tensor = field(init=False, repr=False)
    valid_mask: Tensor = field(init=False, repr=False)
    advanced_mask: Tensor = field(init=False, repr=False)
    step_indices: Tensor = field(init=False, repr=False)


class HaltingBase(Module, HaltingInterface[StateT], Generic[StateT], ABC):
    @classmethod
    def implements_halting_interface(cls) -> bool:
        return (
            issubclass(cls, HaltingInterface)
            and cls.update_halting_state is not HaltingBase.update_halting_state
            and cls.finalize_weighted_accumulation
            is not HaltingBase.finalize_weighted_accumulation
        )

    def update_halting_state(
        self,
        previous_state: StateT | None,
        model_hidden_state: Tensor,
    ) -> tuple[StateT, Tensor]:
        raise NotImplementedError

    def finalize_weighted_accumulation(
        self,
        state: StateT,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
