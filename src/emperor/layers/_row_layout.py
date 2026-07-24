"""Runtime metadata describing rows that were flattened by a semantic owner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False, kw_only=True)
class RowLayout:
    """Describe the semantic axes flattened into a two-dimensional row tensor.

    This value transports row order, padding validity, and a semantic owner's
    decision about whether rows may share generated parameters. It deliberately
    does not infer batch or sequence meaning from tensor rank.
    """

    leading_shape: tuple[int, ...]
    context_sharing_restricted: bool
    batch_axis: int | None = None
    sequence_axis: int | None = None
    valid_rows: Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.__validate_leading_shape()
        self.__validate_restriction_flag()
        self.__normalize_and_validate_axes()
        self.__validate_valid_rows()

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

    def __validate_leading_shape(self) -> None:
        if not isinstance(self.leading_shape, tuple) or not self.leading_shape:
            raise TypeError("leading_shape must be a non-empty tuple.")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.leading_shape
        ):
            raise ValueError(
                "leading_shape dimensions must be positive integers, "
                f"received {self.leading_shape!r}."
            )

    def __validate_restriction_flag(self) -> None:
        if not isinstance(self.context_sharing_restricted, bool):
            raise TypeError(
                "context_sharing_restricted must be a bool, received "
                f"{type(self.context_sharing_restricted).__name__}."
            )

    def __normalize_and_validate_axes(self) -> None:
        if self.batch_axis is None and self.sequence_axis is None:
            if len(self.leading_shape) != 1:
                raise ValueError("row layouts require exactly one leading axis.")
            return
        if self.batch_axis is None or self.sequence_axis is None:
            raise ValueError(
                "sequence layouts require both batch_axis and sequence_axis."
            )
        if len(self.leading_shape) != 2:
            raise ValueError("sequence layouts require exactly two leading axes.")

        normalized_batch_axis = self.__normalize_axis(
            self.batch_axis,
            "batch_axis",
        )
        normalized_sequence_axis = self.__normalize_axis(
            self.sequence_axis,
            "sequence_axis",
        )
        if normalized_batch_axis == normalized_sequence_axis:
            raise ValueError("batch_axis and sequence_axis must be distinct.")
        object.__setattr__(self, "batch_axis", normalized_batch_axis)
        object.__setattr__(self, "sequence_axis", normalized_sequence_axis)

    def __normalize_axis(self, axis: int, name: str) -> int:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"{name} must be an integer, received {axis!r}.")
        axis_count = len(self.leading_shape)
        normalized_axis = axis + axis_count if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= axis_count:
            raise ValueError(
                f"{name} must index leading_shape {self.leading_shape}, "
                f"received {axis}."
            )
        return normalized_axis

    def __validate_valid_rows(self) -> None:
        valid_rows = self.valid_rows
        if valid_rows is None:
            return
        if not isinstance(valid_rows, Tensor):
            raise TypeError(
                "valid_rows must be a Tensor when provided, received "
                f"{type(valid_rows).__name__}."
            )
        if valid_rows.dtype != torch.bool:
            raise TypeError(
                "valid_rows must be a Boolean tensor, received "
                f"dtype={valid_rows.dtype}."
            )
        if valid_rows.dim() != 1:
            raise ValueError(
                "valid_rows must be one-dimensional and aligned with flattened rows, "
                f"received shape {tuple(valid_rows.shape)}."
            )
        if valid_rows.numel() != self.row_count:
            raise ValueError(
                f"valid_rows length must equal row_count={self.row_count}, "
                f"received {valid_rows.numel()}."
            )


__all__ = ("RowLayout",)
