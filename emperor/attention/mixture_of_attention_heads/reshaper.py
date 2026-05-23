from torch import Tensor
from emperor.attention.core._validator import AttentionValidatorBase
from emperor.attention.core.handlers.reshaper import ReshaperBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsReshaper(ReshaperBase):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.top_k: int = cfg.experts_config.top_k
        self.use_kv_expert_models_flag: bool = self.cfg.use_kv_expert_models_flag

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

        query = self.__reshape_q_projection(query, self.qk_head_dim)
        key = self.__reshape_kv_projection(key, static_keys, self.qk_head_dim)
        value = self.__reshape_kv_projection(value, static_values, self.v_head_dim)

        return query, key, value

    def __reshape_q_projection(
        self,
        tensor: Tensor,
        head_dim: int | None = None,
    ) -> Tensor:
        return self.__reshape_projection(tensor, head_dim, is_experts_output=True)

    def __reshape_kv_projection(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
        head_dim: int | None = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor
        return self.__reshape_projection(
            tensor,
            head_dim,
            is_experts_output=self.use_kv_expert_models_flag,
        )

    def __reshape_projection(
        self,
        tensor: Tensor,
        head_dim: int | None = None,
        is_experts_output: bool = False,
    ) -> Tensor:
        head_dim = head_dim or self.head_dim
        shape = self.__get_shape(head_dim, is_experts_output=is_experts_output)
        reshaped_tensor = tensor.view(shape)
        return reshaped_tensor.transpose(0, 1)

    def __get_shape(
        self, head_dim: int, is_experts_output: bool
    ) -> tuple[int, int, int]:
        if is_experts_output:
            return (-1, self.batch_size * self.top_k * self.num_heads, head_dim)
        return (-1, self.batch_size * self.num_heads, head_dim)

    def reshape_before_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self._reshape_query(query)
        key, value = self._reshape_kv(key, value)
        return query, key, value

    def _reshape_query(self, query: Tensor) -> Tensor:
        q_shape = (self.batch_size, self.top_k, self.num_heads, -1, self.qk_head_dim)
        return query.view(q_shape)

    def _reshape_kv(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if self.use_kv_expert_models_flag:
            k_shape = (
                self.batch_size,
                self.top_k,
                self.num_heads,
                -1,
                self.qk_head_dim,
            )
            v_shape = (self.batch_size, self.top_k, self.num_heads, -1, self.v_head_dim)
            return key, value
        k_shape = (self.batch_size, self.num_heads, -1, self.qk_head_dim)
        v_shape = (self.batch_size, self.num_heads, -1, self.v_head_dim)
        return key.view(k_shape), value.view(v_shape)
