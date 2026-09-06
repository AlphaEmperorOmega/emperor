"""Layout and splitting delegate for validated grouping configurations."""

import torch
from torch import Tensor
from torch.nn import functional as F

from emperor.augmentations.adaptive_parameters._grouping.config import GroupingConfig
from emperor.augmentations.adaptive_parameters._grouping.plan import GroupPlan
from emperor.augmentations.adaptive_parameters._grouping.validation import (
    GroupingValidator,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
)


class GroupSplitter:
    """Prepare group members and restoration data without retaining activations."""

    VALIDATOR = GroupingValidator

    def __init__(self, cfg: GroupingConfig):
        self.cfg = cfg
        self.feature_dim = self.cfg.feature_dim
        self.scope = self.cfg.scope
        self.group_count = self.cfg.group_count
        self.chunk_size = self.cfg.chunk_size
        self.sequence_length = self.cfg.sequence_length
        self.input_order = self.cfg.input_order

    def split(self, input_rows: Tensor) -> GroupPlan:
        self.VALIDATOR.validate_flat_input(input_rows, self.cfg)
        canonical, input_order = self.__prepare_grouping_input(input_rows)
        if self.chunk_size is not None:
            return self.__split_by_chunk_size(canonical, input_order)
        return self.__split_by_group_count(canonical, input_order)

    def __prepare_grouping_input(
        self, input_rows: Tensor
    ) -> tuple[Tensor, AdaptiveParameterInputOrderOptions]:
        if self.scope is AdaptiveParameterGroupingScopeOptions.ROWS:
            return self.__prepare_row_input(input_rows)
        return self.__prepare_sequence_input(input_rows)

    def __prepare_row_input(
        self, input_rows: Tensor
    ) -> tuple[Tensor, AdaptiveParameterInputOrderOptions]:
        row_count = input_rows.size(0)
        canonical = input_rows.reshape(1, row_count, self.feature_dim)
        input_order = AdaptiveParameterInputOrderOptions.BATCH_FIRST
        return canonical, input_order

    def __prepare_sequence_input(
        self, input_rows: Tensor
    ) -> tuple[Tensor, AdaptiveParameterInputOrderOptions]:
        row_count = input_rows.size(0)
        batch_size = row_count // self.sequence_length
        input_order = self.input_order
        if input_order is AdaptiveParameterInputOrderOptions.BATCH_FIRST:
            canonical = input_rows.reshape(
                batch_size, self.sequence_length, self.feature_dim
            )
        else:
            sequence_first_tokens = input_rows.reshape(
                self.sequence_length, batch_size, self.feature_dim
            )
            canonical = sequence_first_tokens.transpose(0, 1)
        return canonical, input_order

    def __split_by_chunk_size(
        self, canonical: Tensor, input_order: AdaptiveParameterInputOrderOptions
    ) -> GroupPlan:
        batch_size, sequence_length, feature_dim = canonical.shape
        groups_per_sequence = (sequence_length + self.chunk_size - 1) // self.chunk_size
        padded_sequences, valid_members = self.__pad_sequences(
            canonical, groups_per_sequence
        )
        grouped_shape = (batch_size * groups_per_sequence, self.chunk_size, feature_dim)
        grouped_members = padded_sequences.reshape(grouped_shape)
        return GroupPlan(
            grouped_members=grouped_members,
            canonical_shape=(batch_size, sequence_length),
            input_order=input_order,
            valid_members=valid_members,
        )

    def __pad_sequences(
        self, canonical: Tensor, groups_per_sequence: int
    ) -> tuple[Tensor, Tensor | None]:
        sequence_length = canonical.size(1)
        padded_length = groups_per_sequence * self.chunk_size
        if padded_length == sequence_length:
            return canonical, None

        padded_sequences = self.__pad_sequence_tail(canonical, padded_length)
        valid_members = self.__build_valid_members_mask(
            padded_sequences, sequence_length, groups_per_sequence
        )
        return padded_sequences, valid_members

    def __pad_sequence_tail(self, canonical: Tensor, padded_length: int) -> Tensor:
        sequence_length = canonical.size(1)
        sequence_padding = (0, 0, 0, padded_length - sequence_length)
        padded_sequences = F.pad(canonical, sequence_padding)
        return padded_sequences

    def __build_valid_members_mask(
        self,
        padded_sequences: Tensor,
        sequence_length: int,
        groups_per_sequence: int,
    ) -> Tensor:
        batch_size = padded_sequences.size(0)
        padded_length = padded_sequences.size(1)
        sequence_positions = torch.arange(padded_length, device=padded_sequences.device)
        valid_positions = sequence_positions < sequence_length
        group_mask_shape = (batch_size * groups_per_sequence, self.chunk_size)
        sequence_validity_mask = valid_positions.expand(batch_size, padded_length)
        group_validity_mask = sequence_validity_mask.reshape(group_mask_shape)
        return group_validity_mask

    def __split_by_group_count(
        self, canonical: Tensor, input_order: AdaptiveParameterInputOrderOptions
    ) -> GroupPlan:
        batch_size, sequence_length, feature_dim = canonical.shape
        grouped_shape = (
            batch_size * self.group_count,
            sequence_length // self.group_count,
            feature_dim,
        )
        grouped_members = canonical.reshape(grouped_shape)
        return GroupPlan(
            grouped_members=grouped_members,
            canonical_shape=(batch_size, sequence_length),
            input_order=input_order,
        )
