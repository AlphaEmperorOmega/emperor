import torch
import torch.nn.functional as F

from torch import Tensor
from emperor.attention.core.handlers.processor import ProcessorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.variants.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )
    from emperor.attention.core.variants.mixture_of_attention_heads.projector import (
        MixtureOfAttentionHeadsProjector,
    )
    from emperor.attention.core.handlers.reshaper import ReshaperBase


class MixtureOfAttentionHeadsProcessor(ProcessorBase):
    def __init__(
        self,
        cfg: "MixtureOfAttentionHeadsConfig",
        projector: "MixtureOfAttentionHeadsProjector",
        reshaper: "ReshaperBase",
    ):
        super().__init__(cfg, projector, reshaper)
        self.top_k: int = projector.top_k
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        query, key, value = self.reshaper.reshape_before_attention(query, key, value)
        weights = self.__compute_masked_attention_weights(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(weights, value)
        output = self._compute_attention_output(weighted_value)

        return output, None

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
        total_batch_size = self.batch_size * self.num_heads * self.top_k
        source_sequence_length = key.size(-2)

        key = key.transpose(-2, -1)
        einsum_equation = "bkhie,bhej->bkhij"
        if self.use_kv_expert_models_flag:
            einsum_equation = "bkhie,bkhej->bkhij"
        weights = torch.einsum(einsum_equation, query, key)
        weights = self.__maybe_add_relative_positional_embedding(query, weights)
        weights = weights.contiguous().view(
            total_batch_size, self.target_sequence_length, source_sequence_length
        )
        weights = self.__maybe_add_attention_mask(weights, attention_mask)
        return weights

    def __maybe_add_relative_positional_embedding(
        self, query: Tensor, attention_weights: Tensor
    ) -> Tensor:
        if self.relative_positional_embedding is not None:
            batch_size, top_k, num_heads, target_sequence_length, head_dim = (
                query.size()
            )
            query = query.contiguous().view(
                -1, num_heads, target_sequence_length, head_dim
            )
            source_sequence_length = attention_weights.size(-1)
            positional_embedding = self.relative_positional_embedding(
                query, source_sequence_length
            )
            positional_embedding = positional_embedding.contiguous().view(
                batch_size,
                top_k,
                num_heads,
                target_sequence_length,
                source_sequence_length,
            )

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
    ) -> Tensor:
        source_sequence_length = values.size(-2)
        einsum_equation = "bkhie,bhej->bkhij"
        if self.use_kv_expert_models_flag:
            einsum_equation = "bkhie,bkhej->bkhij"

        attention_weights = attention_weights.contiguous().view(
            self.batch_size,
            self.top_k,
            self.num_heads,
            self.target_sequence_length,
            source_sequence_length,
        )
        weighted_values = torch.einsum(einsum_equation, attention_weights, values)
        values = weighted_values.permute(3, 0, 1, 2, 4)
        values = values.contiguous()
        return values.view(
            self.target_sequence_length,
            self.batch_size,
            self.top_k,
            self.value_projection_dim,
        )
