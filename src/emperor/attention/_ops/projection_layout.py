"""Private attention projection row-layout operations."""

from dataclasses import replace
from typing import cast

import torch
from torch import Tensor

from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._validation import AttentionValidatorBase
from emperor.layers import RowLayout


class ProjectionRowLayoutManager:
    def __init__(self, validator: type[AttentionValidatorBase]) -> None:
        self.__validator = validator

    def attach_projection_row_layout(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> MultiHeadAttentionInputs:
        runtime_layout = attention_inputs.runtime_layout
        self.__validator.validate_projection_row_layout_runtime_layout(runtime_layout)
        runtime_layout = cast(AttentionRuntimeLayout, runtime_layout)

        is_self_attention = self.__is_self_attention(attention_inputs)
        static_source_is_provided = self.__static_source_is_provided(attention_inputs)
        valid_projection_rows = self.__resolve_valid_projection_rows(
            attention_inputs,
            runtime_layout,
            is_self_attention=is_self_attention,
            static_source_is_provided=static_source_is_provided,
        )
        context_sharing_restricted = self.__context_sharing_is_restricted(
            attention_inputs,
            is_self_attention=is_self_attention,
        )
        projection_row_layout = RowLayout.sequence(
            leading_shape=(
                runtime_layout.target_sequence_length,
                runtime_layout.batch_size,
            ),
            batch_axis=1,
            sequence_axis=0,
            valid_rows=valid_projection_rows,
            context_sharing_restricted=context_sharing_restricted,
        )
        return replace(
            attention_inputs,
            runtime_layout=runtime_layout.with_row_layout(projection_row_layout),
        )

    @staticmethod
    def __is_self_attention(attention_inputs: MultiHeadAttentionInputs) -> bool:
        return (
            attention_inputs.query is attention_inputs.key
            and attention_inputs.key is attention_inputs.value
        )

    @staticmethod
    def __static_source_is_provided(
        attention_inputs: MultiHeadAttentionInputs,
    ) -> bool:
        return (
            attention_inputs.static_key is not None
            or attention_inputs.static_value is not None
        )

    @classmethod
    def __resolve_valid_projection_rows(
        cls,
        attention_inputs: MultiHeadAttentionInputs,
        runtime_layout: AttentionRuntimeLayout,
        *,
        is_self_attention: bool,
        static_source_is_provided: bool,
    ) -> Tensor | None:
        if not is_self_attention or static_source_is_provided:
            return None
        return cls.__flatten_valid_self_attention_rows(
            attention_inputs.key_padding_mask,
            runtime_layout,
        )

    @staticmethod
    def __context_sharing_is_restricted(
        attention_inputs: MultiHeadAttentionInputs,
        *,
        is_self_attention: bool,
    ) -> bool:
        return (
            not is_self_attention
            or attention_inputs.static_key is not None
            or attention_inputs.static_value is not None
            or attention_inputs.attention_mask is not None
        )

    @staticmethod
    def __flatten_valid_self_attention_rows(
        key_padding_mask: Tensor | None,
        runtime_layout: AttentionRuntimeLayout,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return None
        expected_shape = (
            runtime_layout.batch_size,
            runtime_layout.source_sequence_length,
        )
        if tuple(key_padding_mask.shape) != expected_shape:
            raise ValueError(
                "Prepared key padding mask must align with attention source rows, "
                f"expected {expected_shape}, received "
                f"{tuple(key_padding_mask.shape)}."
            )
        valid_batch_major_rows = ~torch.isneginf(key_padding_mask)
        return valid_batch_major_rows.transpose(0, 1).reshape(-1)
