"""Private attention batch-dimension operations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._runtime import QKV, AttentionMasks, AttentionRuntimeLayout

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig


class BatchDimensionManager:
    def __init__(self, cfg: MultiHeadAttentionConfig) -> None:
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.batch_first_flag = self.cfg.batch_first_flag

    def convert_inputs_to_internal_layout(
        self,
        qkv: QKV,
        masks: AttentionMasks,
        static_keys: Tensor | None = None,
    ) -> tuple[QKV, AttentionMasks, AttentionRuntimeLayout]:
        input_was_batched = qkv.query.dim() == 3
        input_was_batch_first = self.__input_is_batch_first(qkv.query)
        qkv = self.__maybe_transpose_batch_first_qkv(qkv, input_was_batch_first)
        qkv, masks = self.__maybe_add_batch_dimension_to_unbatched_inputs(
            qkv, masks, input_was_batched
        )

        source_sequence_length = self.__resolve_source_sequence_length(
            qkv.key, static_keys
        )
        runtime_layout = AttentionRuntimeLayout(
            batch_size=qkv.query.size(1),
            target_sequence_length=qkv.query.size(0),
            source_sequence_length=source_sequence_length,
            input_was_batched=input_was_batched,
            input_was_batch_first=input_was_batch_first,
        )
        return qkv, masks, runtime_layout

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
        qkv: QKV,
        input_was_batch_first: bool,
    ) -> QKV:
        if not input_was_batch_first:
            return qkv
        query, key, value = self.__transpose_preserving_shared_tensors(
            qkv.query, qkv.key, qkv.value
        )
        return replace(qkv, query=query, key=key, value=value)

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
        qkv: QKV,
        masks: AttentionMasks,
        input_was_batched: bool,
    ) -> tuple[QKV, AttentionMasks]:
        if input_was_batched:
            return qkv, masks
        query, key, value = self.__unsqueeze_preserving_shared_tensors(
            qkv.query, qkv.key, qkv.value
        )
        qkv = replace(qkv, query=query, key=key, value=value)
        if masks.key_padding_mask is not None:
            key_padding_mask = masks.key_padding_mask.unsqueeze(0)
            masks = replace(
                masks,
                key_padding_mask=key_padding_mask,
            )
        return qkv, masks

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
        runtime_layout: AttentionRuntimeLayout,
    ) -> Tensor:
        if not runtime_layout.input_was_batched:
            attention_output_without_synthetic_batch_dimension = (
                attention_output.squeeze(1)
            )
            return attention_output_without_synthetic_batch_dimension
        if runtime_layout.input_was_batch_first:
            attention_output_in_batch_first_layout = attention_output.transpose(0, 1)
            return attention_output_in_batch_first_layout
        return attention_output
