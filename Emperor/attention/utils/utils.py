import torch
import torch.nn.functional as F

from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig
    from Emperor.attention.utils._validator import MultiHeadAttentionConfigValidator


class Utils:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "MultiHeadAttentionConfigValidator",
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
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()

    def __resolve_qkv_head_dim(self):
        qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if self.query_key_projection_dim != 0
            else self.head_dim
        )
        v_head_dim = (
            self.value_projection_dim // self.num_heads
            if self.value_projection_dim != 0
            else self.head_dim
        )
        return qk_head_dim, v_head_dim

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

    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.check_static_projection_shapes(static_keys, static_values)

        query = self.__reshape_projection_tesnor(query, None, self.qk_head_dim)
        key = self.__reshape_projection_tesnor(key, static_keys, self.qk_head_dim)
        value = self.__reshape_projection_tesnor(value, static_values, self.v_head_dim)

        return query, key, value

    def __reshape_projection_tesnor(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor

        sequence_length = tensor.shape[0]
        head_dim = head_dim or self.head_dim
        shape = (sequence_length, self.batch_size * self.num_heads, head_dim)
        reshaped_tensor = tensor.view(shape)
        return reshaped_tensor.transpose(0, 1)

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
        zero_attetion_shape = (self.batch_size * self.num_heads, 1, self.head_dim)
        zeros_tensor = torch.zeros(
            zero_attetion_shape, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, zeros_tensor], dim=1)

    def merge_padding_and_attention_mask(
        self,
        key: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask

        source_sequence_length = key.size(1)

        shape_view = (self.batch_size, 1, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.view(shape_view)

        shape_expand = (-1, self.num_heads, -1, -1)
        key_padding_mask = key_padding_mask.expand(shape_expand)

        batch_size = self.batch_size * self.num_heads
        shape_reshape = (batch_size, 1, source_sequence_length)
        key_padding_mask = key_padding_mask.reshape(shape_reshape)

        attention_mask = self.__merge_attention_and_padding_mask(
            key_padding_mask, attention_mask
        )
        return attention_mask

    def __merge_attention_and_padding_mask(
        self,
        key_padding_mask: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if attention_mask is None:
            return key_padding_mask
        return attention_mask + key_padding_mask
