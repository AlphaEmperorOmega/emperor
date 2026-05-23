import torch.nn.functional as F

from torch import Tensor
from emperor.attention.core.handlers.processor import ProcessorBase


class IndependentProcessor(ProcessorBase):
    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        attention_mask = self.__prepare_attention_mask(attention_mask)
        query, key, value = self.reshaper.reshape_before_attention(query, key, value)
        weighted_values = self.__compute_weighted_values(
            query, key, value, attention_mask
        )
        attention_output = self._compute_attention_output(weighted_values)
        if self._is_single_batch(attention_output):
            attention_output = attention_output.squeeze(1)
        return attention_output, None

    def __prepare_attention_mask(
        self, attention_mask: Tensor | None = None
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        is_mask_single_batch = attention_mask.size(0) == 1
        is_mask_batched = attention_mask.dim() == 3
        source_sequence_length = attention_mask.size(-1)
        if is_mask_single_batch and is_mask_batched:
            return attention_mask.unsqueeze(0)
        if attention_mask.size(1) == 1:
            return attention_mask.reshape(
                self.batch_size, self.num_heads, 1, source_sequence_length
            )
        return attention_mask.view(
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            source_sequence_length,
        )

    def __compute_weighted_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        weighted_values = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            self.dropout_probability,
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(
            self.batch_size * self.target_sequence_length,
            self.value_projection_dim,
        )
