"""Private independent-attention processing implementation."""

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase

if TYPE_CHECKING:
    from emperor.attention._runtime import QKV, AttentionRuntimeShape


class IndependentProcessor(ProcessorBase):
    def compute_attention(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        merged_attention_mask = self.__prepare_attention_mask(
            merged_attention_mask,
            runtime_shape,
        )
        qkv = self.reshaper.reshape_before_attention(qkv, runtime_shape)
        relative_logits = self._compute_relative_position_logits(
            qkv.query,
            qkv.key.size(qkv.key.dim() - 2),
            runtime_shape,
        )
        if relative_logits is not None:
            if merged_attention_mask is None:
                merged_attention_mask = relative_logits
            else:
                merged_attention_mask = merged_attention_mask + relative_logits
        weighted_values = self.__compute_weighted_values(
            qkv.query,
            qkv.key,
            qkv.value,
            merged_attention_mask,
            runtime_shape,
        )
        attention_output = self._compute_attention_output(
            weighted_values,
            runtime_shape,
        )
        return attention_output, None

    def __prepare_attention_mask(
        self,
        attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else self.target_sequence_length
        )
        is_mask_single_batch = attention_mask.size(0) == 1
        is_mask_batched = attention_mask.dim() == 3
        source_sequence_length = attention_mask.size(-1)
        if is_mask_single_batch and is_mask_batched:
            return attention_mask.reshape(
                1,
                1,
                attention_mask.size(-2),
                source_sequence_length,
            )
        if attention_mask.size(1) == 1:
            return attention_mask.reshape(
                batch_size,
                self.num_heads,
                1,
                source_sequence_length,
            )
        return attention_mask.view(
            batch_size,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )

    def __compute_weighted_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        weighted_values = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            self.dropout_probability if self.training else 0.0,
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else self.target_sequence_length
        )
        return weighted_values.view(
            batch_size * target_sequence_length,
            self.value_projection_dim,
        )
