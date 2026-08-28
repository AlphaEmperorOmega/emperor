from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from torch import Tensor

from emperor.layers._composition.residual.config import ResidualConfig
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState


class ResidualRuntimeRequirement(Enum):
    """Execution requirements declared by a residual connection."""

    FORWARD_LOCAL_STATE = "forward-local residual state"
    DEPTH_SPECIFIC_CONNECTIONS = "depth-specific residual connections"


class ResidualState(ABC):
    """Private marker for residual state scoped to one forward execution."""

    @abstractmethod
    def fork(self) -> ResidualState:
        """Return branch-local state for an independently executed branch."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement branch-local state forking."
        )


class ResidualStateLifecycle(ABC):
    """Create residual state scoped to one forward execution."""

    @abstractmethod
    def create_state(self, initial_source: Tensor) -> ResidualState | None:
        """Create state from the source entering the execution owner."""


class ResidualConnectionAbstract(Module, ABC):
    """Stable runtime Interface implemented by every residual variant."""

    VALIDATOR = ResidualConnectionValidator
    supports_pairwise_diagnostics: ClassVar[bool] = False
    RUNTIME_REQUIREMENTS: ClassVar[frozenset[ResidualRuntimeRequirement]] = frozenset()

    def __init__(
        self,
        cfg: ResidualConfig,
        overrides: ResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: ResidualConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.residual_dim: int | None = self.cfg.residual_dim

    @property
    def residual_state_lifecycle(self) -> ResidualStateLifecycle | None:
        return None

    def new_state(self, initial_source: Tensor) -> ResidualState | None:
        lifecycle = self.residual_state_lifecycle
        if lifecycle is None:
            return None
        return lifecycle.create_state(initial_source)

    def apply_to_layer_state(
        self,
        state: LayerState,
        previous: Tensor,
    ) -> LayerState:
        """Apply this connection using residual context carried by a LayerState."""
        state.hidden = self(
            state.hidden,
            previous,
            residual_state=state.residual_state,
            row_layout=state.row_layout,
        )
        return state

    @abstractmethod
    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: ResidualState | None = None,
        row_layout: RowLayout | None = None,
    ) -> Tensor:
        """Compose current and previous sources."""
