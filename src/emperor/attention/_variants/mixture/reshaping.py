"""Private mixture-of-attention-heads reshaping implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._ops.reshaping import ReshaperBase

if TYPE_CHECKING:
    from emperor.attention._runtime import QKV, AttentionRuntimeShape
    from emperor.attention._variants.mixture.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsReshaper(ReshaperBase):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.top_k: int = cfg.experts_config.top_k
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

    def reshape_qkv_for_attention(
        self,
        qkv: "QKV",
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> "QKV":
        if self.use_kv_expert_models_flag and (
            static_keys is not None or static_values is not None
        ):
            raise ValueError(
                "static key/value projections are not supported when "
                "use_kv_expert_models_flag is True."
            )
        self.VALIDATOR.validate_static_projection_shapes(
            self, static_keys, static_values, runtime_shape
        )

        query = self.__reshape_q_projection(
            qkv.query,
            self.qk_head_dim,
            runtime_shape,
        )
        key = self.__reshape_kv_projection(
            qkv.key, static_keys, self.qk_head_dim, runtime_shape
        )
        value = self.__reshape_kv_projection(
            qkv.value, static_values, self.v_head_dim, runtime_shape
        )

        return replace(qkv, query=query, key=key, value=value)

    def __reshape_q_projection(
        self,
        tensor: Tensor,
        head_dim: int | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        return self.__reshape_projection(
            tensor, head_dim, is_experts_output=True, runtime_shape=runtime_shape
        )

    def __reshape_kv_projection(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor
        return self.__reshape_projection(
            tensor,
            head_dim,
            is_experts_output=self.use_kv_expert_models_flag,
            runtime_shape=runtime_shape,
        )

    def __reshape_projection(
        self,
        tensor: Tensor,
        head_dim: int | None = None,
        *,
        is_experts_output: bool,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        head_dim = head_dim or self.head_dim
        shape = self.__get_shape(
            head_dim,
            is_experts_output=is_experts_output,
            runtime_shape=runtime_shape,
        )
        reshaped_tensor = tensor.view(shape)
        return reshaped_tensor.transpose(0, 1)

    def __get_shape(
        self,
        head_dim: int,
        is_experts_output: bool,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[int, int, int]:
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        if is_experts_output:
            return (-1, batch_size * self.top_k * self.num_heads, head_dim)
        return (-1, batch_size * self.num_heads, head_dim)

    def reshape_before_attention(
        self,
        qkv: "QKV",
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> "QKV":
        query = self._reshape_query(qkv.query, runtime_shape)
        key, value = self._reshape_kv(qkv.key, qkv.value, runtime_shape)
        return replace(qkv, query=query, key=key, value=value)

    def _reshape_query(
        self, query: Tensor, runtime_shape: "AttentionRuntimeShape | None" = None
    ) -> Tensor:
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        q_shape = (batch_size, self.top_k, self.num_heads, -1, self.qk_head_dim)
        return query.view(q_shape)

    def _reshape_kv(
        self,
        key: Tensor,
        value: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        if self.use_kv_expert_models_flag:
            k_shape = (
                batch_size,
                self.top_k,
                self.num_heads,
                -1,
                self.qk_head_dim,
            )
            v_shape = (batch_size, self.top_k, self.num_heads, -1, self.v_head_dim)
            return key.view(k_shape), value.view(v_shape)
        k_shape = (batch_size, self.num_heads, -1, self.qk_head_dim)
        v_shape = (batch_size, self.num_heads, -1, self.v_head_dim)
        return key.view(k_shape), value.view(v_shape)
