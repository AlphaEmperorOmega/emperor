"""Runtime metadata describing rows that were flattened by a semantic owner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

from torch import Tensor

from emperor.layers._row_layout.validation import RowLayoutValidator


@dataclass(frozen=True, eq=False, kw_only=True)
class RowLayout:
    """Describe the semantic axes flattened into a two-dimensional row tensor.

    This value transports row order, padding validity, and a semantic owner's
    decision about whether rows may share generated parameters. It deliberately
    does not infer batch or sequence meaning from tensor rank.
    """

    VALIDATOR = RowLayoutValidator

    leading_shape: tuple[int, ...]
    context_sharing_restricted: bool
    batch_axis: int | None = None
    sequence_axis: int | None = None
    valid_rows: Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_batch_axis, normalized_sequence_axis = self.VALIDATOR.validate(self)
        object.__setattr__(self, "batch_axis", normalized_batch_axis)
        object.__setattr__(self, "sequence_axis", normalized_sequence_axis)

    @classmethod
    def rows(
        cls,
        row_count: int,
        *,
        context_sharing_restricted: bool,
        valid_rows: Tensor | None = None,
    ) -> RowLayout:
        return cls(
            leading_shape=(row_count,),
            context_sharing_restricted=context_sharing_restricted,
            valid_rows=valid_rows,
        )

    @classmethod
    def sequence(
        cls,
        *,
        leading_shape: tuple[int, int],
        batch_axis: int,
        sequence_axis: int,
        context_sharing_restricted: bool,
        valid_rows: Tensor | None = None,
    ) -> RowLayout:
        return cls(
            leading_shape=leading_shape,
            batch_axis=batch_axis,
            sequence_axis=sequence_axis,
            valid_rows=valid_rows,
            context_sharing_restricted=context_sharing_restricted,
        )

    @property
    def row_count(self) -> int:
        return prod(self.leading_shape)

    @property
    def is_sequence(self) -> bool:
        return self.sequence_axis is not None

    def with_context_sharing_restricted(self) -> RowLayout:
        if self.context_sharing_restricted:
            return self
        return replace(self, context_sharing_restricted=True)


__all__ = ("RowLayout",)
