"""Private self-attention processing implementation."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase

if TYPE_CHECKING:
    from emperor.attention._runtime import QKV, AttentionRuntimeShape


class SelfAttentionProcessor(ProcessorBase):
    def compute_attention(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__compute_masked_attention_weights(
            qkv.query,
            qkv.key,
            merged_attention_mask,
            runtime_shape,
        )
        weighted_value = self.__compute_weighted_values(
            weights,
            qkv.value,
            runtime_shape,
        )
        output = self._compute_attention_output(weighted_value, runtime_shape)
        weights = self.__format_attention_weights(weights, runtime_shape)

        return output, weights

    def __compute_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        scaled_query = self.__scale_query(query)
        raw_weights = self.__compute_raw_masked_attention_weights(
            scaled_query, key, attention_mask, runtime_shape
        )
        weights = F.softmax(raw_weights, dim=-1)
        return F.dropout(
            weights,
            p=self.dropout_probability,
            training=self.training,
        )

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        key = key.transpose(-2, -1)
        weights = torch.bmm(query, key)
        weights = self.__maybe_add_relative_positional_embedding(
            query, weights, runtime_shape
        )
        weights = self.__maybe_add_attention_mask(weights, attention_mask)
        return weights

    def __maybe_add_relative_positional_embedding(
        self,
        query: Tensor,
        attention_weights: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        positional_embedding = self._compute_relative_position_logits(
            query,
            attention_weights.size(-1),
            runtime_shape,
            query_is_scaled=True,
        )
        if positional_embedding is not None:
            return positional_embedding + attention_weights
        return attention_weights

    def __maybe_add_attention_mask(
        self, weights: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        if attention_mask is not None:
            return weights + attention_mask
        return weights

    def __compute_weighted_values(
        self,
        attention_weights: Tensor,
        values: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        weighted_values = torch.bmm(attention_weights, values)
        values = weighted_values.transpose(0, 1)
        values = values.contiguous()
        target_sequence_length = values.size(0)
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        return values.view(
            target_sequence_length * batch_size,
            self.value_projection_dim,
        )

    def __format_attention_weights(
        self,
        attention_weights: Tensor,
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> Tensor | None:
        if not self.return_attention_weights_flag:
            return None

        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else self.target_sequence_length
        )
        source_sequence_length = attention_weights.size(-1)
        attention_weights_shape = (
            batch_size,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )
        attention_weights = attention_weights.view(attention_weights_shape)
        attention_weights = self.__maybe_average_attention_weights(attention_weights)

        if runtime_shape is not None and not runtime_shape.input_was_batched:
            return attention_weights.squeeze(0)
        return attention_weights

    def __maybe_average_attention_weights(self, attention_weights: Tensor) -> Tensor:
        if self.average_attention_weights_flag:
            return attention_weights.mean(dim=1)
        return attention_weights

    def __ensure_correct_shape_output(
        self,
        attention_output: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if not self.return_attention_weights_flag:
            return attention_output, None
        formatted_weights = self.__format_attention_weights(
            attention_weights,
            None,
        )
        return self.__handle_batched_input(attention_output, formatted_weights)

    def __handle_batched_input(
        self,
        attention_output: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if attention_output.dim() == 3 and attention_output.size(1) == 1:
            return attention_output.squeeze(1), attention_weights.squeeze(0)
        return attention_output, attention_weights
