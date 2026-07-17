"""Private self-attention processing implementation."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase
from emperor.attention._runtime import QKV

if TYPE_CHECKING:
    from emperor.attention._runtime import AttentionRuntimeLayout


class SelfAttentionProcessor(ProcessorBase):
    def compute_attention(
        self,
        qkv: QKV,
        merged_attention_mask: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        attention_weights = self.__compute_masked_attention_weights(
            qkv, merged_attention_mask, runtime_layout
        )
        weighted_values = self.__compute_weighted_values(
            qkv, attention_weights, runtime_layout
        )
        attention_output = self._compute_attention_output(
            weighted_values, runtime_layout
        )
        attention_weights = self.__format_attention_weights(
            attention_weights, runtime_layout
        )

        return attention_output, attention_weights

    def __compute_masked_attention_weights(
        self,
        qkv: QKV,
        merged_attention_mask: Tensor | None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        scaled_query = self.__scale_query(qkv.query)
        scaled_qkv = QKV(query=scaled_query, key=qkv.key, value=qkv.value)
        raw_attention_weights = self.__compute_raw_masked_attention_weights(
            scaled_qkv, merged_attention_mask, runtime_layout
        )
        fully_masked_rows = torch.isneginf(raw_attention_weights).all(
            dim=-1, keepdim=True
        )
        safe_raw_attention_weights = raw_attention_weights.masked_fill(
            fully_masked_rows, 0.0
        )
        attention_weights = F.softmax(safe_raw_attention_weights, dim=-1)
        attention_weights = attention_weights.masked_fill(fully_masked_rows, 0.0)
        return F.dropout(
            attention_weights, p=self.dropout_probability, training=self.training
        )

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        qkv: QKV,
        merged_attention_mask: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        transposed_key = qkv.key.transpose(-2, -1)
        attention_weights = torch.bmm(qkv.query, transposed_key)
        attention_weights = self.__add_relative_position_logits_if_available(
            qkv,
            attention_weights,
            runtime_layout,
        )
        attention_weights = self.__add_attention_mask_if_available(
            attention_weights,
            merged_attention_mask,
        )
        return attention_weights

    def __add_relative_position_logits_if_available(
        self,
        qkv: QKV,
        attention_weights: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        relative_position_logits = self.__compute_relative_position_logits_for_qkv(
            qkv,
            runtime_layout,
        )
        if relative_position_logits is not None:
            return relative_position_logits + attention_weights
        return attention_weights

    def __compute_relative_position_logits_for_qkv(
        self,
        qkv: QKV,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor | None:
        source_sequence_dimension = qkv.key.dim() - 2
        source_sequence_length = qkv.key.size(source_sequence_dimension)
        return self._compute_relative_position_logits(
            qkv.query,
            source_sequence_length,
            runtime_layout,
            query_is_scaled=True,
        )

    def __add_attention_mask_if_available(
        self,
        attention_weights: Tensor,
        merged_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if merged_attention_mask is not None:
            return attention_weights + merged_attention_mask
        return attention_weights

    def __compute_weighted_values(
        self,
        qkv: QKV,
        attention_weights: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        weighted_values = torch.bmm(attention_weights, qkv.value)
        weighted_values = weighted_values.transpose(0, 1)
        weighted_values = weighted_values.contiguous()
        target_sequence_length = weighted_values.size(0)
        batch_size = self.__resolve_batch_size(runtime_layout)
        return weighted_values.view(
            target_sequence_length * batch_size,
            self.value_projection_dim,
        )

    def __resolve_batch_size(
        self,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> int:
        if runtime_layout is not None:
            return runtime_layout.batch_size
        return self.batch_size

    def __format_attention_weights(
        self,
        attention_weights: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> Tensor | None:
        if not self.return_attention_weights_flag:
            return None

        batch_size = self.__resolve_batch_size(runtime_layout)
        target_sequence_length = self.__resolve_target_sequence_length(runtime_layout)
        source_sequence_length = attention_weights.size(-1)
        attention_weights_shape = (
            batch_size,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )
        attention_weights = attention_weights.view(attention_weights_shape)
        attention_weights = self.__maybe_average_attention_weights(attention_weights)

        if runtime_layout is not None and not runtime_layout.input_was_batched:
            return attention_weights.squeeze(0)
        return attention_weights

    def __resolve_target_sequence_length(
        self,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> int:
        if runtime_layout is not None:
            return runtime_layout.target_sequence_length
        return self.target_sequence_length

    def __maybe_average_attention_weights(self, attention_weights: Tensor) -> Tensor:
        if self.average_attention_weights_flag:
            return attention_weights.mean(dim=1)
        return attention_weights
