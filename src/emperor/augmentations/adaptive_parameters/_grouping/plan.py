"""Forward-local grouping and restoration for an all-valid flat input."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from emperor.augmentations.adaptive_parameters._grouping.validation import (
    GroupingValidator,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterInputOrderOptions,
)


@dataclass(frozen=True, eq=False)
class GroupPlan:
    VALIDATOR = GroupingValidator

    grouped_members: Tensor
    canonical_shape: tuple[int, int]
    input_order: AdaptiveParameterInputOrderOptions
    valid_members: Tensor | None = None

    @property
    def context_count(self) -> int:
        return self.grouped_members.size(0)

    @property
    def members_per_group(self) -> int:
        return self.grouped_members.size(1)

    @property
    def row_count(self) -> int:
        return self.canonical_shape[0] * self.canonical_shape[1]

    @property
    def physical_row_count(self) -> int:
        return self.context_count * self.members_per_group

    def restore(
        self, grouped_output: Tensor, *, output_dim: int | None = None
    ) -> Tensor:
        expected_leading_shape = (self.context_count, self.members_per_group)
        self.VALIDATOR.validate_grouped_output(
            grouped_output, expected_leading_shape, output_dim
        )
        batch_size, sequence_length = self.canonical_shape
        output_dim = grouped_output.size(-1)
        restored = grouped_output.reshape(batch_size, -1, output_dim)
        restored = restored[:, :sequence_length]
        if self.input_order is AdaptiveParameterInputOrderOptions.SEQUENCE_FIRST:
            restored = restored.transpose(0, 1)
        return restored.reshape(self.row_count, output_dim)
