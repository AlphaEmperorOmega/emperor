from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention.core.runtime import QKV, AttentionMasks, AttentionRuntimeShape

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig


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
    ) -> tuple[QKV, AttentionMasks, AttentionRuntimeShape]:
        input_was_batched = qkv.query.dim() == 3
        input_was_batch_first = self.__input_is_batch_first(qkv.query)
        qkv = self.__maybe_transpose_batch_first_qkv(qkv, input_was_batch_first)
        qkv, masks = self.__maybe_add_batch_dimension_to_unbatched_inputs(
            qkv, masks, input_was_batched
        )

        source_sequence_length = self.__resolve_source_sequence_length(
            qkv.key, static_keys
        )
        runtime_shape = AttentionRuntimeShape(
            batch_size=qkv.query.size(1),
            target_sequence_length=qkv.query.size(0),
            source_sequence_length=source_sequence_length,
            input_was_batched=input_was_batched,
            input_was_batch_first=input_was_batch_first,
        )
        return qkv, masks, runtime_shape

    def __input_is_batch_first(self, query: Tensor) -> bool:
        if query.dim() != 3:
            return False
        if self.batch_first_flag is not None:
            return self.batch_first_flag
        # Historical behavior inferred layout by asking whether dimension 1 was the
        # configured batch size. Explicit flags avoid this ambiguity for new models.
        return query.size(1) != self.batch_size

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
        runtime_shape: AttentionRuntimeShape,
    ) -> Tensor:
        if not runtime_shape.input_was_batched:
            return attention_output.squeeze(1)
        if runtime_shape.input_was_batch_first:
            return attention_output.transpose(0, 1)
        return attention_output
