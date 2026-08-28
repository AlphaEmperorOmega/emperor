from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.layers._row_layout.core import RowLayout


class RowLayoutValidator(ValidatorBase):
    @classmethod
    def validate(
        cls,
        row_layout: RowLayout,
    ) -> tuple[int | None, int | None]:
        leading_shape = cls._validate_leading_shape(row_layout.leading_shape)
        cls._validate_restriction_flag(row_layout.context_sharing_restricted)
        normalized_axes = cls._normalize_and_validate_axes(
            leading_shape,
            row_layout.batch_axis,
            row_layout.sequence_axis,
        )
        cls._validate_valid_rows(
            row_layout.valid_rows,
            row_count=row_layout.row_count,
        )
        return normalized_axes

    @staticmethod
    def _validate_leading_shape(leading_shape: object) -> tuple[int, ...]:
        if not isinstance(leading_shape, tuple) or not leading_shape:
            raise TypeError("leading_shape must be a non-empty tuple.")
        candidate_shape = cast(tuple[object, ...], leading_shape)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in candidate_shape
        ):
            raise ValueError(
                "leading_shape dimensions must be positive integers, "
                f"received {candidate_shape!r}."
            )
        return cast(tuple[int, ...], candidate_shape)

    @staticmethod
    def _validate_restriction_flag(context_sharing_restricted: object) -> None:
        if not isinstance(context_sharing_restricted, bool):
            raise TypeError(
                "context_sharing_restricted must be a bool, received "
                f"{type(context_sharing_restricted).__name__}."
            )

    @classmethod
    def _normalize_and_validate_axes(
        cls,
        leading_shape: tuple[int, ...],
        batch_axis: object,
        sequence_axis: object,
    ) -> tuple[int | None, int | None]:
        if batch_axis is None and sequence_axis is None:
            if len(leading_shape) != 1:
                raise ValueError("row layouts require exactly one leading axis.")
            return None, None
        if batch_axis is None or sequence_axis is None:
            raise ValueError(
                "sequence layouts require both batch_axis and sequence_axis."
            )
        if len(leading_shape) != 2:
            raise ValueError("sequence layouts require exactly two leading axes.")

        normalized_batch_axis = cls._normalize_axis(
            batch_axis,
            "batch_axis",
            leading_shape,
        )
        normalized_sequence_axis = cls._normalize_axis(
            sequence_axis,
            "sequence_axis",
            leading_shape,
        )
        if normalized_batch_axis == normalized_sequence_axis:
            raise ValueError("batch_axis and sequence_axis must be distinct.")
        return normalized_batch_axis, normalized_sequence_axis

    @staticmethod
    def _normalize_axis(
        axis: object,
        name: str,
        leading_shape: tuple[int, ...],
    ) -> int:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"{name} must be an integer, received {axis!r}.")
        axis_count = len(leading_shape)
        normalized_axis = axis + axis_count if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= axis_count:
            raise ValueError(
                f"{name} must index leading_shape {leading_shape}, received {axis}."
            )
        return normalized_axis

    @staticmethod
    def _validate_valid_rows(valid_rows: object, *, row_count: int) -> None:
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
        if valid_rows.numel() != row_count:
            raise ValueError(
                f"valid_rows length must equal row_count={row_count}, "
                f"received {valid_rows.numel()}."
            )
