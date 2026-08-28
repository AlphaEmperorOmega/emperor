from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingStateBase
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._row_layout import RowLayout


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: HaltingStateBase | None = None
    residual_state: ResidualState | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )
    row_layout: RowLayout | None = field(
        default=None,
        kw_only=True,
        repr=False,
        compare=False,
    )

    @contextmanager
    def scoped_residual_state(
        self, replacement: ResidualState | None
    ) -> Iterator[None]:
        enclosing_residual_state = self.residual_state
        self.residual_state = replacement
        try:
            yield
        finally:
            self.residual_state = enclosing_residual_state
