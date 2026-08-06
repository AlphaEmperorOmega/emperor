from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch.nn as nn
from torch import Tensor

from emperor.layers._composition.recurrent.validation import (
    RecurrentResidualScheduleValidator,
)
from emperor.layers._composition.residual.base import (
    ResidualConnectionAbstract,
    ResidualRuntimeRequirement,
    ResidualState,
)

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout


class RecurrentResidualSchedule(nn.Module, ABC):
    """Select and apply residual connections across recurrent transitions."""

    VALIDATOR = RecurrentResidualScheduleValidator

    def __init__(self, transition_count: int) -> None:
        super().__init__()
        self.transition_count = transition_count
        self.VALIDATOR.validate(self)

    @staticmethod
    def new_state(
        primary_connection: ResidualConnectionAbstract,
        initial_source: Tensor,
    ) -> ResidualState | None:
        return primary_connection.new_state(initial_source)

    def apply(
        self,
        primary_connection: ResidualConnectionAbstract,
        transition_index: int,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
        row_layout: RowLayout | None = None,
    ) -> Tensor:
        connection = self.connection_for_transition(
            primary_connection,
            transition_index,
        )
        return connection(
            current,
            previous,
            residual_state=residual_state,
            row_layout=row_layout,
        )

    def connection_for_transition(
        self,
        primary_connection: ResidualConnectionAbstract,
        transition_index: int,
    ) -> ResidualConnectionAbstract:
        self.VALIDATOR.validate_transition_index(self, transition_index)
        return self._connection_for_transition(
            primary_connection,
            transition_index,
        )

    @abstractmethod
    def _connection_for_transition(
        self,
        primary_connection: ResidualConnectionAbstract,
        transition_index: int,
    ) -> ResidualConnectionAbstract:
        """Return the connection assigned to one recurrent transition."""


class SharedRecurrentResidualSchedule(RecurrentResidualSchedule):
    """Reuse the primary residual connection at every recurrent transition."""

    def _connection_for_transition(
        self,
        primary_connection: ResidualConnectionAbstract,
        transition_index: int,
    ) -> ResidualConnectionAbstract:
        return primary_connection


class DepthwiseRecurrentResidualSchedule(RecurrentResidualSchedule):
    """Own one independent residual connection per recurrent transition."""

    def __init__(
        self,
        transition_count: int,
        subsequent_connections: Iterable[ResidualConnectionAbstract],
    ) -> None:
        super().__init__(transition_count)
        self.subsequent_connections = nn.ModuleList(subsequent_connections)
        self.VALIDATOR.validate_subsequent_connections(self)

    @classmethod
    def from_connection(
        cls,
        primary_connection: ResidualConnectionAbstract,
        transition_count: int,
    ) -> DepthwiseRecurrentResidualSchedule:
        return cls(
            transition_count,
            (
                cast(
                    ResidualConnectionAbstract,
                    copy.deepcopy(primary_connection.cfg).build(),
                )
                for _ in range(transition_count - 1)
            ),
        )

    def _connection_for_transition(
        self,
        primary_connection: ResidualConnectionAbstract,
        transition_index: int,
    ) -> ResidualConnectionAbstract:
        if transition_index == 0:
            return primary_connection
        return cast(
            ResidualConnectionAbstract,
            self.subsequent_connections[transition_index - 1],
        )


def build_recurrent_residual_schedule(
    primary_connection: ResidualConnectionAbstract | None,
    transition_count: int,
) -> RecurrentResidualSchedule | None:
    if primary_connection is None:
        return None
    if (
        ResidualRuntimeRequirement.DEPTH_SPECIFIC_CONNECTIONS
        in primary_connection.RUNTIME_REQUIREMENTS
    ):
        return DepthwiseRecurrentResidualSchedule.from_connection(
            primary_connection,
            transition_count,
        )
    return SharedRecurrentResidualSchedule(transition_count)
