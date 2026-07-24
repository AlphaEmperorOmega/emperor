"""Compact grouping primitives for adaptive parameter generation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
)
from emperor.layers import RowLayout


@dataclass(frozen=True, eq=False)
class AdaptiveGroupPlan:
    grouped_members: Tensor
    valid_members: Tensor | None
    leading_shape: tuple[int, ...]
    canonical_shape: tuple[int, int]
    inverse_permutation: tuple[int, int, int]

    @property
    def context_count(self) -> int:
        return self.grouped_members.size(0)

    @property
    def members_per_group(self) -> int:
        return self.grouped_members.size(1)

    @property
    def row_count(self) -> int:
        return self.context_count * self.members_per_group

    def restore(self, grouped_output: Tensor) -> Tensor:
        if not isinstance(grouped_output, Tensor):
            raise TypeError(
                "grouped_output must be a Tensor, received "
                f"{type(grouped_output).__name__}."
            )
        if grouped_output.dim() != 3:
            raise ValueError(
                "grouped_output must have shape "
                "(context_count, members_per_group, output_dim), received "
                f"{tuple(grouped_output.shape)}."
            )
        expected_leading_shape = (self.context_count, self.members_per_group)
        if tuple(grouped_output.shape[:2]) != expected_leading_shape:
            raise ValueError(
                "grouped_output leading dimensions must equal "
                f"{expected_leading_shape}, received {tuple(grouped_output.shape[:2])}."
            )

        batch_size, sequence_length = self.canonical_shape
        output_dim = grouped_output.size(-1)
        canonical_output = grouped_output.reshape(
            batch_size,
            sequence_length,
            output_dim,
        )
        original_layout_output = canonical_output.permute(self.inverse_permutation)
        return original_layout_output.reshape(self.row_count, output_dim)


def build_adaptive_group_plan(
    input_rows: Tensor,
    grouping_scope: AdaptiveParameterGroupingScopeOptions,
    group_count: int,
    row_layout: RowLayout,
) -> AdaptiveGroupPlan:
    _validate_common_inputs(input_rows, grouping_scope, group_count, row_layout)
    if grouping_scope == AdaptiveParameterGroupingScopeOptions.DISABLED:
        raise ValueError("Cannot build an adaptive group plan for DISABLED grouping.")
    if grouping_scope == AdaptiveParameterGroupingScopeOptions.ROWS:
        return _build_rows_plan(input_rows, group_count, row_layout)
    if grouping_scope == AdaptiveParameterGroupingScopeOptions.SEQUENCE:
        return _build_sequence_plan(input_rows, group_count, row_layout)
    raise ValueError(
        f"Unsupported adaptive parameter grouping scope {grouping_scope!r}."
    )


def _validate_common_inputs(
    input_rows: Tensor,
    grouping_scope: AdaptiveParameterGroupingScopeOptions,
    group_count: int,
    row_layout: RowLayout,
) -> None:
    if not isinstance(input_rows, Tensor):
        raise TypeError(
            f"input_rows must be a Tensor, received {type(input_rows).__name__}."
        )
    if input_rows.dim() != 2:
        raise ValueError(
            "input_rows must be a two-dimensional matrix, received "
            f"shape {tuple(input_rows.shape)}."
        )
    if not isinstance(grouping_scope, AdaptiveParameterGroupingScopeOptions):
        raise TypeError(
            "grouping_scope must be an AdaptiveParameterGroupingScopeOptions value, "
            f"received {grouping_scope!r}."
        )
    if not isinstance(row_layout, RowLayout):
        raise TypeError(
            f"row_layout must be a RowLayout, received {type(row_layout).__name__}."
        )
    if row_layout.row_count != input_rows.size(0):
        raise ValueError(
            f"row_layout row_count={row_layout.row_count} does not match input row "
            f"count {input_rows.size(0)}."
        )
    if row_layout.valid_rows is not None and (
        row_layout.valid_rows.device != input_rows.device
    ):
        raise ValueError(
            "row_layout.valid_rows must be on the same device as input_rows, "
            f"received {row_layout.valid_rows.device} and {input_rows.device}."
        )
    if row_layout.context_sharing_restricted:
        raise ValueError(
            "Adaptive parameter grouping cannot run because context sharing is "
            "restricted by the semantic owner."
        )
    _validate_enabled_group_count(group_count)


def _validate_enabled_group_count(group_count: int) -> None:
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count <= 0
    ):
        raise ValueError(
            "group_count must be a positive integer for enabled grouping, "
            f"received {group_count!r}."
        )


def _build_rows_plan(
    input_rows: Tensor,
    group_count: int,
    row_layout: RowLayout,
) -> AdaptiveGroupPlan:
    if row_layout.is_sequence:
        raise ValueError("ROWS grouping requires a one-axis row layout.")
    row_count = input_rows.size(0)
    _validate_divisibility(row_count, group_count, "row count")
    members_per_group = row_count // group_count
    grouped_members = input_rows.reshape(
        group_count,
        members_per_group,
        input_rows.size(-1),
    )
    grouped_valid_rows = _group_valid_rows(
        row_layout.valid_rows,
        group_count,
        members_per_group,
    )
    return AdaptiveGroupPlan(
        grouped_members=grouped_members,
        valid_members=grouped_valid_rows,
        leading_shape=row_layout.leading_shape,
        canonical_shape=(1, row_count),
        inverse_permutation=(0, 1, 2),
    )


def _build_sequence_plan(
    input_rows: Tensor,
    group_count: int,
    row_layout: RowLayout,
) -> AdaptiveGroupPlan:
    if not row_layout.is_sequence:
        raise ValueError("SEQUENCE grouping requires a two-axis sequence layout.")
    batch_axis = row_layout.batch_axis
    sequence_axis = row_layout.sequence_axis
    batch_size = row_layout.leading_shape[batch_axis]
    sequence_length = row_layout.leading_shape[sequence_axis]
    _validate_divisibility(sequence_length, group_count, "sequence length")
    members_per_group = sequence_length // group_count
    feature_dim = input_rows.size(-1)
    input_in_original_layout = input_rows.reshape(
        *row_layout.leading_shape, feature_dim
    )
    to_canonical_permutation = (batch_axis, sequence_axis, 2)
    canonical_input = input_in_original_layout.permute(to_canonical_permutation)
    grouped_members = canonical_input.reshape(
        batch_size * group_count,
        members_per_group,
        feature_dim,
    )

    grouped_valid_rows = None
    if row_layout.valid_rows is not None:
        valid_in_original_layout = row_layout.valid_rows.reshape(
            *row_layout.leading_shape
        )
        canonical_valid_rows = valid_in_original_layout.permute(
            batch_axis,
            sequence_axis,
        )
        grouped_valid_rows = canonical_valid_rows.reshape(
            batch_size * group_count,
            members_per_group,
        )
    return AdaptiveGroupPlan(
        grouped_members=grouped_members,
        valid_members=grouped_valid_rows,
        leading_shape=row_layout.leading_shape,
        canonical_shape=(batch_size, sequence_length),
        inverse_permutation=_invert_permutation(to_canonical_permutation),
    )


def _validate_divisibility(
    axis_length: int,
    group_count: int,
    axis_name: str,
) -> None:
    if group_count > axis_length:
        raise ValueError(
            f"group_count={group_count} cannot exceed {axis_name} {axis_length}."
        )
    if axis_length % group_count != 0:
        raise ValueError(
            f"{axis_name} {axis_length} must be divisible by group_count={group_count}."
        )


def _group_valid_rows(
    valid_rows: Tensor | None,
    group_count: int,
    members_per_group: int,
) -> Tensor | None:
    if valid_rows is None:
        return None
    return valid_rows.reshape(group_count, members_per_group)


def _invert_permutation(permutation: tuple[int, int, int]) -> tuple[int, int, int]:
    inverse = [0, 0, 0]
    for canonical_axis, original_axis in enumerate(permutation):
        inverse[original_axis] = canonical_axis
    return tuple(inverse)
