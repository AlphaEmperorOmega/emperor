from torch import Tensor
from emperor.base.utils import Module
from emperor.attention.utils.handlers.validators._reshaper import ReshaperValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.utils.layer import MultiHeadAttentionConfig


class ReshaperBuilder:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        self.cfg = cfg
        self.attention_option = self.cfg.attention_option

    def build(self) -> "ReshaperBase":
        from emperor.attention.utils.enums import AttentionOptions

        match self.attention_option:
            case AttentionOptions.MIXTURE_OF_ATTENTION_HEADS:
                return MixtureOfAttentionHeadsReshaper(self.cfg)
            case _:
                return AttentionReshaper(self.cfg)


class ReshaperBase:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        self.cfg = cfg
        self.batch_size: int = self.cfg.batch_size
        self.num_heads: int = self.cfg.num_heads
        self.embedding_dim: int = self.cfg.embedding_dim
        self.target_sequence_length: int = self.cfg.target_sequence_length
        self.attention_option = self.cfg.attention_option
        self.query_key_projection_dim: int = self.cfg.query_key_projection_dim
        self.value_projection_dim: int = self.cfg.value_projection_dim
        self.head_dim = self.embedding_dim // self.num_heads
        self.qk_head_dim, self.v_head_dim = self.__resolve_qkv_head_dim()
        self.validator = ReshaperValidator(self)

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

    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError(
            "`reshape_qkv_for_attention` method must be implemented by subclass"
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
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        super().__init__(cfg)

    def reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.check_static_projection_shapes(static_keys, static_values)

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
        self.validator.ensure_not_self_attention()
        query = self._reshape_query(query)
        key, value = self._reshape_kv(key, value)
        return query, key, value


class MixtureOfAttentionHeadsReshaper(ReshaperBase):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
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
        self.validator.check_static_projection_shapes(static_keys, static_values)

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

    def __get_shape(self, head_dim: int, is_experts_output: bool):
        if is_experts_output:
            return (-1, self.batch_size * self.top_k * self.num_heads, head_dim)
        return (-1, self.batch_size * self.num_heads, head_dim)

    def reshape_before_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.ensure_not_self_attention()
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
            v_shape = (
                self.batch_size,
                self.top_k,
                self.num_heads,
                -1,
                self.v_head_dim,
            )
            return key, value
        k_shape = (
            self.batch_size,
            self.num_heads,
            -1,
            self.qk_head_dim,
        )
        v_shape = (
            self.batch_size,
            self.num_heads,
            -1,
            self.v_head_dim,
        )
        return key.view(k_shape), value.view(v_shape)
