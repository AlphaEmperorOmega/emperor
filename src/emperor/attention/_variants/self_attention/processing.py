"""Private self-attention processing implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase

if TYPE_CHECKING:
    from emperor.attention._runtime import (
        AttentionRuntimeLayout,
        MultiHeadAttentionInputs,
    )


class SelfAttentionProcessor(ProcessorBase):
    def compute_attention(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> tuple[Tensor, Tensor | None]:
        runtime_layout = attention_inputs.runtime_layout
        attention_weights = self._compute_masked_attention_weights(attention_inputs)
        weighted_values = self.__compute_weighted_values(
            attention_inputs,
            attention_weights,
        )
        attention_output = self._compute_attention_output(
            weighted_values, runtime_layout
        )
        attention_weights = self.__format_attention_weights(
            attention_weights, runtime_layout
        )

        return attention_output, attention_weights

    def _compute_masked_attention_weights(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> Tensor:
        scaled_inputs = replace(
            attention_inputs,
            query=self.__scale_query(attention_inputs.query),
        )
        raw_attention_weights = self.__compute_raw_masked_attention_weights(
            scaled_inputs
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
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> Tensor:
        transposed_key = attention_inputs.key.transpose(-2, -1)
        attention_weights = torch.bmm(attention_inputs.query, transposed_key)
        attention_weights = self.__add_relative_position_logits_if_available(
            attention_inputs,
            attention_weights,
        )
        attention_weights = self.__add_attention_mask_if_available(
            attention_weights,
            attention_inputs.merged_attention_mask,
        )
        return attention_weights

    def __add_relative_position_logits_if_available(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
        attention_weights: Tensor,
    ) -> Tensor:
        relative_position_logits = self.__compute_relative_position_logits_for_inputs(
            attention_inputs
        )
        if relative_position_logits is not None:
            return relative_position_logits + attention_weights
        return attention_weights

    def __compute_relative_position_logits_for_inputs(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> Tensor | None:
        source_sequence_dimension = attention_inputs.key.dim() - 2
        source_sequence_length = attention_inputs.key.size(source_sequence_dimension)
        return self._compute_relative_position_logits(
            attention_inputs.query,
            source_sequence_length,
            attention_inputs.runtime_layout,
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
        attention_inputs: "MultiHeadAttentionInputs",
        attention_weights: Tensor,
    ) -> Tensor:
        weighted_values = torch.bmm(attention_weights, attention_inputs.value)
        weighted_values = weighted_values.transpose(0, 1)
        weighted_values = weighted_values.contiguous()
        target_sequence_length = weighted_values.size(0)
        batch_size = self.__resolve_batch_size(attention_inputs.runtime_layout)
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
