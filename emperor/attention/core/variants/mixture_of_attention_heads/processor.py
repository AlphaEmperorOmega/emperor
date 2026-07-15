from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention.core.handlers.processor import ProcessorBase
from emperor.attention.core.variants.mixture_of_attention_heads.validator import (
    MixtureOfAttentionHeadsValidator,
)

if TYPE_CHECKING:
    from emperor.attention.core.handlers.reshaper import ReshaperBase
    from emperor.attention.core.runtime import QKV, AttentionRuntimeShape
    from emperor.attention.core.variants.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )
    from emperor.attention.core.variants.mixture_of_attention_heads.projector import (
        MixtureOfAttentionHeadsProjector,
    )


class MixtureOfAttentionHeadsProcessor(ProcessorBase):
    VALIDATOR = MixtureOfAttentionHeadsValidator

    def __init__(
        self,
        cfg: "MixtureOfAttentionHeadsConfig",
        projector: "MixtureOfAttentionHeadsProjector",
        reshaper: "ReshaperBase",
    ):
        super().__init__(cfg, projector, reshaper)
        self.top_k: int = projector.top_k
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

    def _compute_relative_position_logits(
        self,
        query: Tensor,
        source_sequence_length: int,
        runtime_shape: "AttentionRuntimeShape | None" = None,
        *,
        query_is_scaled: bool = False,
    ) -> Tensor | None:
        if self.relative_positional_embedding is None:
            return None
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        expected_leading_shape = (batch_size, self.top_k, self.num_heads)
        self.VALIDATOR.validate_relative_position_query_shape(
            query,
            expected_leading_shape,
        )

        target_sequence_length = query.size(-2)
        head_width = query.size(-1)
        flattened_query = query.contiguous().view(
            batch_size * self.top_k,
            self.num_heads,
            target_sequence_length,
            head_width,
        )
        relative_position_logits = super()._compute_relative_position_logits(
            flattened_query,
            source_sequence_length,
            runtime_shape,
            query_is_scaled=query_is_scaled,
        )
        flattened_logits = cast(Tensor, relative_position_logits)
        return flattened_logits.contiguous().view(
            batch_size,
            self.top_k,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )

    def compute_attention(
        self,
        qkv: "QKV",
        merged_attention_mask: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[Tensor, Tensor | None]:
        qkv = self.reshaper.reshape_before_attention(qkv, runtime_shape)
        weights = self.__compute_masked_attention_weights(
            qkv.query, qkv.key, merged_attention_mask, runtime_shape
        )
        weighted_value = self.__compute_weighted_values(
            weights, qkv.value, runtime_shape
        )
        output = self._compute_attention_output(weighted_value, runtime_shape)

        return output, None

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
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else self.target_sequence_length
        )
        total_batch_size = batch_size * self.num_heads * self.top_k
        source_sequence_length = key.size(-2)

        key = key.transpose(-2, -1)
        if self.use_kv_expert_models_flag:
            weights = torch.matmul(query, key)
        else:
            weights = torch.matmul(query, key.unsqueeze(1))
        weights = self.__maybe_add_relative_positional_embedding(
            query,
            weights,
            runtime_shape,
        )
        weights = weights.contiguous().view(
            total_batch_size, target_sequence_length, source_sequence_length
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
            attention_weights = attention_weights + positional_embedding
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
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        target_sequence_length = (
            runtime_shape.target_sequence_length
            if runtime_shape is not None
            else self.target_sequence_length
        )
        source_sequence_length = values.size(-2)
        if self.use_kv_expert_models_flag:
            values_for_attention = values
        else:
            values_for_attention = values.unsqueeze(1)

        attention_weights = attention_weights.contiguous().view(
            batch_size,
            self.top_k,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )
        weighted_values = torch.matmul(attention_weights, values_for_attention)
        values = weighted_values.permute(3, 0, 1, 2, 4)
        values = values.contiguous()
        return values.view(
            target_sequence_length, batch_size, self.top_k, self.value_projection_dim
        )
