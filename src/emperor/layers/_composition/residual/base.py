from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from torch import Tensor

from emperor.layers._composition.residual.config import ResidualConfig
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._row_layout import RowLayout


class ResidualState:
    """Private marker for residual state scoped to one forward execution."""


class ResidualConnectionAbstract(Module, ABC):
    """Stable runtime Interface implemented by every residual variant."""

    VALIDATOR = ResidualConnectionValidator
    supports_pairwise_diagnostics: ClassVar[bool] = False

    def __init__(
        self,
        cfg: ResidualConfig,
        overrides: ResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: ResidualConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.residual_dim: int | None = self.cfg.residual_dim

    def new_state(self, initial_source: Tensor) -> ResidualState | None:
        return None

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
