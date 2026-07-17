"""Private independent-attention processing implementation."""

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase

if TYPE_CHECKING:
    from emperor.attention._runtime import QKV, AttentionRuntimeLayout


class IndependentProcessor(ProcessorBase):
    def compute_attention(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        qkv = self.reshaper.reshape_before_attention(qkv, runtime_layout)
        weighted_values = self.__compute_weighted_values(
            qkv, merged_attention_mask, runtime_layout
        )
        attention_output = self._compute_attention_output(
            weighted_values, runtime_layout
        )
        return attention_output, None

    def __compute_weighted_values(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        effective_attention_mask = self.__prepare_effective_attention_mask(
            qkv,
            merged_attention_mask,
            runtime_layout,
        )
        dropout_probability = self.dropout_probability if self.training else 0.0
        weighted_values = F.scaled_dot_product_attention(
            qkv.query,
            qkv.key,
            qkv.value,
            effective_attention_mask,
            dropout_probability,
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        batch_size = self.__resolve_batch_size(runtime_layout)
        target_sequence_length = self.__resolve_target_sequence_length(runtime_layout)
        return weighted_values.view(
            batch_size * target_sequence_length,
            self.value_projection_dim,
        )

    def __prepare_effective_attention_mask(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> Tensor | None:
        effective_attention_mask = self.__prepare_attention_mask(
            merged_attention_mask,
            runtime_layout,
        )
        relative_position_logits = self.__compute_relative_position_logits_for_qkv(
            qkv,
            runtime_layout,
        )
        if relative_position_logits is None:
            return effective_attention_mask
        if effective_attention_mask is None:
            return relative_position_logits
        return effective_attention_mask + relative_position_logits

    def __prepare_attention_mask(
        self,
        attention_mask: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask
        is_mask_single_batch = attention_mask.size(0) == 1
        is_mask_batched = attention_mask.dim() == 3
        if is_mask_single_batch and is_mask_batched:
            return self.__reshape_mask_for_batch_head_broadcasting(attention_mask)
        if attention_mask.size(1) == 1:
            return self.__reshape_mask_for_target_sequence_broadcasting(
                attention_mask, runtime_layout
            )
        return self.__reshape_mask_to_batch_head_target_source_layout(
            attention_mask, runtime_layout
        )

    def __reshape_mask_for_batch_head_broadcasting(
        self,
        attention_mask: Tensor,
    ) -> Tensor:
        target_sequence_length = attention_mask.size(-2)
        source_sequence_length = attention_mask.size(-1)
        return attention_mask.reshape(
            1, 1, target_sequence_length, source_sequence_length
        )

    def __reshape_mask_for_target_sequence_broadcasting(
        self,
        attention_mask: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> Tensor:
        batch_size = self.__resolve_batch_size(runtime_layout)
        source_sequence_length = attention_mask.size(-1)
        return attention_mask.reshape(
            batch_size, self.num_heads, 1, source_sequence_length
        )

    def __reshape_mask_to_batch_head_target_source_layout(
        self,
        attention_mask: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> Tensor:
        batch_size = self.__resolve_batch_size(runtime_layout)
        target_sequence_length = self.__resolve_target_sequence_length(runtime_layout)
        source_sequence_length = attention_mask.size(-1)
        return attention_mask.view(
            batch_size, self.num_heads, target_sequence_length, source_sequence_length
        )

    def __compute_relative_position_logits_for_qkv(
        self,
        qkv: "QKV",
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor | None:
        source_sequence_dimension = qkv.key.dim() - 2
        source_sequence_length = qkv.key.size(source_sequence_dimension)
        return self._compute_relative_position_logits(
            qkv.query, source_sequence_length, runtime_layout
        )

    def __resolve_batch_size(
        self,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> int:
        if runtime_layout is not None:
            return runtime_layout.batch_size
        return self.batch_size

    def __resolve_target_sequence_length(
        self,
        runtime_layout: "AttentionRuntimeLayout | None",
    ) -> int:
        if runtime_layout is not None:
            return runtime_layout.target_sequence_length
        return self.target_sequence_length
