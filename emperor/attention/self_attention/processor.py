import torch
import torch.nn.functional as F

from torch import Tensor
from emperor.attention.core.handlers.processor import ProcessorBase


class SelfAttentionProcessor(ProcessorBase):
    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        weights = self.__compute_masked_attention_weights(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(weights, value)
        output = self._compute_attention_output(weighted_value)
        output, weights = self.__ensure_correct_shape_output(output, weights)

        return output, weights

    def __compute_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        scaled_query = self.__scale_query(query)
        raw_weights = self.__compute_raw_masked_attention_weights(
            scaled_query, key, attention_mask
        )
        weights = F.softmax(raw_weights, dim=-1)
        if self.dropout_probability > 0.0:
            weights = F.dropout(
                weights, p=self.dropout_probability, training=self.training
            )
        return weights

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        key = key.transpose(-2, -1)
        weights = torch.bmm(query, key)
        weights = self.__maybe_add_relative_positional_embedding(query, weights)
        weights = self.__maybe_add_attention_mask(weights, attention_mask)
        return weights

    def __maybe_add_relative_positional_embedding(
        self, query: Tensor, attention_weights: Tensor
    ) -> Tensor:
        if self.relative_positional_embedding is not None:
            query = query.contiguous().view(
                self.batch_size, self.num_heads, -1, self.qk_head_dim
            )
            target_sequence_length = query.size(-2)
            positional_embedding = self.relative_positional_embedding(
                query, sequence_length=target_sequence_length
            )
            source_sequence_length = positional_embedding.size(-1)
            positional_embedding = positional_embedding.contiguous().view(
                self.batch_size * self.num_heads,
                target_sequence_length,
                source_sequence_length,
            )
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
    ) -> Tensor:
        weighted_values = torch.bmm(attention_weights, values)
        values = weighted_values.transpose(0, 1)
        values = values.contiguous()
        target_sequence_length = values.size(0)
        return values.view(
            target_sequence_length * self.batch_size,
            self.embedding_dim,
        )

    def __ensure_correct_shape_output(
        self,
        attention_output: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if not self.return_attention_weights_flag:
            if self._is_single_batch(attention_output):
                return attention_output, None
            return attention_output.squeeze(1), None

        source_sequence_length = attention_weights.size(-1)
        attention_weights_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            source_sequence_length,
        )
        attention_weights = attention_weights.view(attention_weights_shape)
        attention_weights = self.__maybe_average_attention_weights(attention_weights)

        return self.__handle_batched_input(attention_output, attention_weights)

    def __maybe_average_attention_weights(self, attention_weights: Tensor) -> Tensor:
        if self.average_attention_weights_flag:
            return attention_weights.mean(dim=1)
        return attention_weights

    def __handle_batched_input(
        self, attention_output: Tensor, attention_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self._is_single_batch(attention_output):
            return attention_output, attention_weights
        output_with_removed_batch_dim = attention_output.squeeze(1)
        weights_with_removed_batch_dim = attention_weights.squeeze(0)
        return output_with_removed_batch_dim, weights_with_removed_batch_dim
