from torch import Tensor
from emperor.attention.core._validator import AttentionValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig


class ReshaperBase:
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        self.cfg = cfg
        self.batch_size: int = self.cfg.batch_size
        self.num_heads: int = self.cfg.num_heads
        self.embedding_dim: int = self.cfg.embedding_dim
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        self.head_dim = self.embedding_dim // self.num_heads
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()

    def __resolve_qkv_head_dim(self) -> tuple[int, int]:
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

    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError(
            "reshape_qkv_for_attention must be implemented by subclass."
        )

    def reshape_before_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return query, key, value

    def _reshape_query(self, query: Tensor) -> Tensor:
        q_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.qk_head_dim,
        )
        return query.view(q_shape)

    def _reshape_kv(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        source_sequence_length = key.size(1)
        k_shape = (
            self.batch_size,
            self.num_heads,
            source_sequence_length,
            self.qk_head_dim,
        )
        v_shape = (
            self.batch_size,
            self.num_heads,
            source_sequence_length,
            self.v_head_dim,
        )
        return key.view(k_shape), value.view(v_shape)


class AttentionReshaper(ReshaperBase):
    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        AttentionValidatorBase.validate_static_projection_shapes(
            self, static_keys, static_values
        )

        query = self.__reshape_projection(query, None, self.qk_head_dim)
        key = self.__reshape_projection(key, static_keys, self.qk_head_dim)
        value = self.__reshape_projection(value, static_values, self.v_head_dim)

        return query, key, value

    def __reshape_projection(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor

        sequence_length = tensor.size(0)
        head_dim = head_dim or self.head_dim
        shape = (sequence_length, self.batch_size * self.num_heads, head_dim)
        reshaped_tensor = tensor.view(shape)
        return reshaped_tensor.transpose(0, 1)

    def reshape_before_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self._reshape_query(query)
        key, value = self._reshape_kv(key, value)
        return query, key, value
