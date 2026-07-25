"""Private attention batch-dimension operations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from torch import Tensor

from emperor.attention._runtime import (
    AttentionRuntimeLayout,
    MultiHeadAttentionInputs,
)
from emperor.attention._validation import AttentionValidatorBase

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig


class BatchDimensionManager:
    VALIDATOR = AttentionValidatorBase

    def __init__(self, cfg: MultiHeadAttentionConfig) -> None:
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.batch_first_flag = self.cfg.batch_first_flag

    def convert_inputs_to_internal_layout(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> MultiHeadAttentionInputs:
        query = attention_inputs.query
        key = attention_inputs.key
        value = attention_inputs.value
        key_padding_mask = attention_inputs.key_padding_mask
        input_was_batched = query.dim() == 3
        input_was_batch_first = self.__input_is_batch_first(query)
        query, key, value = self.__maybe_transpose_batch_first_qkv(
            query,
            key,
            value,
            input_was_batch_first,
        )
        query, key, value, key_padding_mask = (
            self.__maybe_add_batch_dimension_to_unbatched_inputs(
                query,
                key,
                value,
                key_padding_mask,
                input_was_batched,
            )
        )

        source_sequence_length = self.__resolve_source_sequence_length(
            key, attention_inputs.static_key
        )
        runtime_layout = AttentionRuntimeLayout(
            batch_size=query.size(1),
            target_sequence_length=query.size(0),
            source_sequence_length=source_sequence_length,
            input_was_batched=input_was_batched,
            input_was_batch_first=input_was_batch_first,
        )
        return replace(
            attention_inputs,
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            runtime_layout=runtime_layout,
        )

    def __input_is_batch_first(self, query: Tensor) -> bool:
        query_has_no_explicit_batch_dimension = query.dim() != 3
        if query_has_no_explicit_batch_dimension:
            return False
        if self.batch_first_flag is not None:
            return self.batch_first_flag
        # Historical behavior inferred layout by asking whether dimension 1 was the
        # configured batch size. Explicit flags avoid this ambiguity for new models.
        legacy_layout_is_inferred_as_batch_first = query.size(1) != self.batch_size
        return legacy_layout_is_inferred_as_batch_first

    def __maybe_transpose_batch_first_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        input_was_batch_first: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not input_was_batch_first:
            return query, key, value
        return self.__transpose_preserving_shared_tensors(query, key, value)

    def __transpose_preserving_shared_tensors(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        transposed_query = query.transpose(0, 1)
        transposed_key = transposed_query if key is query else key.transpose(0, 1)
        if value is query:
            transposed_value = transposed_query
        elif value is key:
            transposed_value = transposed_key
        else:
            transposed_value = value.transpose(0, 1)
        return transposed_query, transposed_key, transposed_value

    def __maybe_add_batch_dimension_to_unbatched_inputs(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None,
        input_was_batched: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if input_was_batched:
            return query, key, value, key_padding_mask
        query, key, value = self.__unsqueeze_preserving_shared_tensors(
            query, key, value
        )
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        return query, key, value, key_padding_mask

    def __unsqueeze_preserving_shared_tensors(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batched_query = query.unsqueeze(1)
        batched_key = batched_query if key is query else key.unsqueeze(1)
        if value is query:
            batched_value = batched_query
        elif value is key:
            batched_value = batched_key
        else:
            batched_value = value.unsqueeze(1)
        return batched_query, batched_key, batched_value

    def __resolve_source_sequence_length(
        self,
        key: Tensor,
        static_keys: Tensor | None,
    ) -> int:
        if static_keys is not None and static_keys.dim() == 3:
            return static_keys.size(1)
        return key.size(0)

    def restore_output_layout(
        self,
        attention_output: Tensor,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> Tensor:
        runtime_layout = attention_inputs.runtime_layout
        self.VALIDATOR.validate_output_layout_restoration_runtime_layout(runtime_layout)
        runtime_layout = cast(AttentionRuntimeLayout, runtime_layout)
        if not runtime_layout.input_was_batched:
            attention_output_without_synthetic_batch_dimension = (
                attention_output.squeeze(1)
            )
            return attention_output_without_synthetic_batch_dimension
        if runtime_layout.input_was_batch_first:
            attention_output_in_batch_first_layout = attention_output.transpose(0, 1)
            return attention_output_in_batch_first_layout
        return attention_output
