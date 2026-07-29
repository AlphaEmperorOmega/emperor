"""Private mixture-of-attention-heads processing implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._ops.processing import ProcessorBase
from emperor.attention._variants.mixture.validation import (
    MixtureOfAttentionHeadsValidator,
)

if TYPE_CHECKING:
    from emperor.attention._ops.reshaping import ReshaperBase
    from emperor.attention._runtime import (
        AttentionRuntimeLayout,
        MultiHeadAttentionInputs,
    )
    from emperor.attention._variants.mixture.config import (
        MixtureOfAttentionHeadsConfig,
    )
    from emperor.attention._variants.mixture.projection import (
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

    def compute_attention(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> tuple[Tensor, Tensor | None]:
        attention_inputs = self.reshaper.reshape_before_attention(attention_inputs)
        runtime_layout = attention_inputs.runtime_layout
        attention_weights = self._compute_masked_attention_weights(attention_inputs)
        weighted_values = self.__compute_weighted_values(
            attention_inputs,
            attention_weights,
        )
        attention_output = self._compute_attention_output(
            weighted_values,
            runtime_layout,
        )

        return attention_output, None

    def _compute_masked_attention_weights(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> Tensor:
        scaled_inputs = replace(
            attention_inputs,
            query=self.__scale_query(attention_inputs.query),
        )
        raw_attention_weights = self.__compute_raw_masked_attention_weights(
            scaled_inputs,
        )
        fully_masked_rows = torch.isneginf(raw_attention_weights).all(
            dim=-1,
            keepdim=True,
        )
        safe_raw_attention_weights = raw_attention_weights.masked_fill(
            fully_masked_rows,
            0.0,
        )
        attention_weights = F.softmax(safe_raw_attention_weights, dim=-1)
        attention_weights = attention_weights.masked_fill(fully_masked_rows, 0.0)
        return F.dropout(
            attention_weights,
            p=self.dropout_probability,
            training=self.training,
        )

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        return query * head_dim**-0.5

    def __compute_raw_masked_attention_weights(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
    ) -> Tensor:
        runtime_layout = attention_inputs.runtime_layout
        batch_size = self.__resolve_batch_size(runtime_layout)
        target_sequence_length = self.__resolve_target_sequence_length(runtime_layout)
        total_batch_size = batch_size * self.num_heads * self.top_k
        source_sequence_length = attention_inputs.key.size(-2)

        transposed_key = attention_inputs.key.transpose(-2, -1)
        if self.use_kv_expert_models_flag:
            attention_weights = torch.matmul(attention_inputs.query, transposed_key)
        else:
            attention_weights = torch.matmul(
                attention_inputs.query,
                transposed_key.unsqueeze(1),
            )
        attention_weights = self.__add_relative_position_logits_if_available(
            attention_inputs,
            attention_weights,
        )
        attention_weights = attention_weights.contiguous().view(
            total_batch_size, target_sequence_length, source_sequence_length
        )
        return self.__add_attention_mask_if_available(
            attention_weights,
            attention_inputs.merged_attention_mask,
        )

    def __add_relative_position_logits_if_available(
        self,
        attention_inputs: "MultiHeadAttentionInputs",
        attention_weights: Tensor,
    ) -> Tensor:
        relative_position_logits = self.__compute_relative_position_logits_for_inputs(
            attention_inputs
        )
        if relative_position_logits is not None:
            attention_weights = attention_weights + relative_position_logits
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

    def _compute_relative_position_logits(
        self,
        query: Tensor,
        source_sequence_length: int,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
        *,
        query_is_scaled: bool = False,
    ) -> Tensor | None:
        if self.relative_positional_embedding is None:
            return None
        batch_size = self.__resolve_batch_size(runtime_layout)
        expected_leading_shape = (batch_size, self.top_k, self.num_heads)
        self.VALIDATOR.validate_relative_position_query_shape(
            query,
            expected_leading_shape,
        )

        target_sequence_length = query.size(-2)
        head_width = query.size(-1)
        flattened_query = query.contiguous().view(
            batch_size * self.top_k, self.num_heads, target_sequence_length, head_width
        )
        relative_position_logits = super()._compute_relative_position_logits(
            flattened_query,
            source_sequence_length,
            runtime_layout,
            query_is_scaled=query_is_scaled,
        )
        return relative_position_logits.contiguous().view(
            batch_size,
            self.top_k,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
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
        runtime_layout = attention_inputs.runtime_layout
        batch_size = self.__resolve_batch_size(runtime_layout)
        target_sequence_length = self.__resolve_target_sequence_length(runtime_layout)
        source_sequence_length = attention_inputs.value.size(-2)
        if self.use_kv_expert_models_flag:
            values_for_attention = attention_inputs.value
        else:
            values_for_attention = attention_inputs.value.unsqueeze(1)

        attention_weights = attention_weights.contiguous().view(
            batch_size,
            self.top_k,
            self.num_heads,
            target_sequence_length,
            source_sequence_length,
        )
        weighted_values = torch.matmul(attention_weights, values_for_attention)
        weighted_values = weighted_values.permute(3, 0, 1, 2, 4)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(
            target_sequence_length, batch_size, self.top_k, self.value_projection_dim
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
