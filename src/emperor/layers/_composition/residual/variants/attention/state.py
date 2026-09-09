from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from torch import Tensor

from emperor.layers._composition.residual.base import ResidualState
from emperor.layers._composition.residual.validation import AttentionResidualValidator


@dataclass(slots=True)
class AttentionResidualState(ResidualState):
    """Forward-local sources mixed once per physical residual-depth execution."""

    VALIDATOR: ClassVar[type[AttentionResidualValidator]] = AttentionResidualValidator

    initial_source: Tensor
    block_size: int
    _completed_blocks: list[Tensor] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _partial_block: Tensor | None = field(default=None, init=False, repr=False)
    _partial_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.VALIDATOR.validate_positive_integer(
            self.block_size,
            name="block_size",
        )

    @property
    def sources(self) -> tuple[Tensor, ...]:
        partial_sources = () if self._partial_block is None else (self._partial_block,)
        return (self.initial_source, *self._completed_blocks, *partial_sources)

    def append(self, raw_output: Tensor) -> None:
        self._partial_block = (
            raw_output
            if self._partial_block is None
            else self._partial_block + raw_output
        )
        self._partial_count += 1
        if self._partial_count == self.block_size:
            self._completed_blocks.append(self._partial_block)
            self._partial_block = None
            self._partial_count = 0

    def fork(self) -> AttentionResidualState:
        forked = AttentionResidualState(
            self.initial_source,
            block_size=self.block_size,
        )
        forked._completed_blocks = list(self._completed_blocks)
        forked._partial_block = self._partial_block
        forked._partial_count = self._partial_count
        return forked
