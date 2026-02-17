import torch
import torch.nn.functional as F

from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig
    from Emperor.attention.utils._validator import MultiHeadAttentionValidator


class Utils:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionValidator",
    ):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.validator = validator
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.head_dim = self.embedding_dim // self.num_heads

    def add_batch_dimension_if_missing(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.validator.is_tensor_batched(query):
            return query, key, value, key_padding_mask, attention_mask
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
        return query, key, value, key_padding_mask, attention_mask

    def add_zero_attention(
        self,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.zero_attention_flag:
            return key, value, key_padding_mask, attention_mask

        padded_key = self.__concatenate_zeros_tensor(key)
        padded_value = self.__concatenate_zeros_tensor(value)
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return padded_key, padded_value, key_padding_mask, attention_mask

    def __concatenate_zeros_tensor(self, tensor: Tensor) -> Tensor:
        head_dim = tensor.size(-1)
        zero_attetion_shape = (self.batch_size * self.num_heads, 1, head_dim)
        zeros_tensor = torch.zeros(
            zero_attetion_shape, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, zeros_tensor], dim=1)
