"""Private attention reshaping operations."""

from dataclasses import replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._validation import AttentionValidatorBase

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import QKV, AttentionRuntimeLayout


class ReshaperBase:
    VALIDATOR = AttentionValidatorBase

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
        has_explicit_query_key_projection_dimension = self.query_key_projection_dim != 0
        qk_head_dim = (
            self.query_key_projection_dim // self.num_heads
            if has_explicit_query_key_projection_dimension
            else self.head_dim
        )
        has_explicit_value_projection_dimension = self.value_projection_dim != 0
        v_head_dim = (
            self.value_projection_dim // self.num_heads
            if has_explicit_value_projection_dimension
            else self.head_dim
        )
        return qk_head_dim, v_head_dim

    def reshape_qkv_for_attention(
        self,
        qkv: "QKV",
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "QKV":
        raise NotImplementedError(
            "reshape_qkv_for_attention must be implemented by subclass."
        )

    def reshape_before_attention(
        self,
        qkv: "QKV",
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "QKV":
        return qkv

    def _reshape_query(
        self, query: Tensor, runtime_layout: "AttentionRuntimeLayout | None" = None
    ) -> Tensor:
        is_runtime_layout_available = runtime_layout is not None
        batch_size = (
            runtime_layout.batch_size
            if is_runtime_layout_available
            else self.batch_size
        )
        target_sequence_length = (
            runtime_layout.target_sequence_length
            if is_runtime_layout_available
            else self.target_sequence_length
        )
        q_shape = (
            batch_size,
            self.num_heads,
            target_sequence_length,
            self.qk_head_dim,
        )
        return query.view(q_shape)

    def _reshape_kv(
        self,
        key: Tensor,
        value: Tensor,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = (
            runtime_layout.batch_size if runtime_layout is not None else self.batch_size
        )
        source_sequence_length = key.size(1)
        k_shape = (batch_size, self.num_heads, source_sequence_length, self.qk_head_dim)
        v_shape = (batch_size, self.num_heads, source_sequence_length, self.v_head_dim)
        return key.view(k_shape), value.view(v_shape)


class AttentionReshaper(ReshaperBase):
    def reshape_qkv_for_attention(
        self,
        qkv: "QKV",
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "QKV":
        self.VALIDATOR.validate_static_projection_shapes(
            self, static_keys, static_values, runtime_layout
        )

        query = self.__reshape_projection(
            qkv.query, None, self.qk_head_dim, runtime_layout
        )
        key = self.__reshape_projection(
            qkv.key, static_keys, self.qk_head_dim, runtime_layout
        )
        value = self.__reshape_projection(
            qkv.value, static_values, self.v_head_dim, runtime_layout
        )

        return replace(qkv, query=query, key=key, value=value)

    def __reshape_projection(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor

        sequence_length = tensor.size(0)
        head_dim = head_dim or self.head_dim
        batch_size = (
            runtime_layout.batch_size if runtime_layout is not None else self.batch_size
        )
        shape = (sequence_length, batch_size * self.num_heads, head_dim)
        reshaped_tensor = tensor.reshape(shape)
        return reshaped_tensor.transpose(0, 1)

    def reshape_before_attention(
        self,
        qkv: "QKV",
        runtime_layout: "AttentionRuntimeLayout | None" = None,
    ) -> "QKV":
        query = self._reshape_query(qkv.query, runtime_layout)
        key, value = self._reshape_kv(qkv.key, qkv.value, runtime_layout)
        return replace(qkv, query=query, key=key, value=value)
