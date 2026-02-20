from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.utils.layer import MultiHeadAttentionConfig


class MultiHeadAttentionValidator:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
    ):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_heads = self.cfg.num_heads
        self.embeding_dim = self.cfg.embedding_dim
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.head_dim = self.embeding_dim // self.num_heads
        self.batched_input_flag = None
        self.attention_option = self.cfg.attention_option
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

    def assert_correct_head_dim(self, head_dim: int) -> None:
        assert (head_dim * self.num_heads) == self.embeding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def check_attention_input_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> bool:
        self.batched_input_flag = self.is_tensor_batched(query)

        self.__check_query_dims(query)
        self.__check_query_key_value_dimension_count(key, value)
        self.__check_key_padding_mask_dimension_count(key_padding_mask)
        self.__check_attention_mask_dim_count_and_shape(attention_mask)

        return self.batched_input_flag

    def __check_query_dims(self, query: Tensor) -> None:
        if query.dim() not in (2, 3):
            raise RuntimeError(
                f"Query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D tensor"
            )

    def __check_query_key_value_dimension_count(
        self, key: Tensor, value: Tensor
    ) -> None:
        expected_dims = 3 if self.batched_input_flag else 2
        is_qk_dim_count_same = key.dim() == expected_dims
        is_qv_dim_count_same = value.dim() == expected_dims
        are_qkv_dim_counts_same = is_qk_dim_count_same and is_qv_dim_count_same
        if not are_qkv_dim_counts_same:
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected key and value to be {expected_dims}-D "
                f"but found {key.dim()}-D and {value.dim()}-D tensors respectively"
            )

    def __check_key_padding_mask_dimension_count(
        self,
        key_padding_mask: Tensor | None = None,
    ) -> None:
        if key_padding_mask is None:
            return

        expected_dim = 2 if self.batched_input_flag else 1
        if key_padding_mask.dim() != expected_dim:
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected `key_padding_mask` to be None or {expected_dim}-D "
                f"but found {key_padding_mask.dim()}-D tensor instead"
            )

    def __check_attention_mask_dim_count_and_shape(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if attention_mask is None:
            return

        if attention_mask.dim() not in (2, 3):
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected attention_mask to be None, 2-D, or 3-D tensor "
                f"but found {attention_mask.dim()}-D tensor instead"
            )

    def __format_dimension_context(self) -> str:
        return "batched (3-D)" if self.batched_input_flag else "unbatched (2-D)"

    def is_tensor_batched(self, tensor: Tensor) -> bool:
        self.batched_input_flag = tensor.dim() == 3
        return self.batched_input_flag

    def get_batched_input_flag(self) -> bool:
        return self.batched_input_flag

    def is_input_batched(self, tensor: Tensor | None = None) -> bool:
        if self.batched_input_flag is None:
            assert tensor is not None, (
                "Tensor must be provided to check batch dimension."
            )
            self.batched_input_flag = tensor.dim() == 3
            return self.batched_input_flag
        return self.batched_input_flag

    def assert_correct_embedding_dim(self, expected_embedding_dim: int):
        assert self.embedding_dim == expected_embedding_dim, (
            f"Was expecting embedding dimension of {expected_embedding_dim}, but got {self.embedding_dim}"
        )

    def are_separate_projection_models_initialized(self) -> None:
        ensure_qkv_models_exist = (
            self.query_model is not None
            and self.key_model is not None
            and self.value_model is not None
        )
        assert ensure_qkv_models_exist, (
            "When query, key, and value are not the same and self attention is not performed, ensure `attention_option` is `True`"
        )

    def validate_attention_weights_flag_with_projection_type(self):
        assert not self.return_attention_weights_flag, (
            "`attention_weights` can be returned only when self attention is performed, ensure that `attention_option` is set to `False` and the `query`, `key` and `value` tensors are the same tensor."
        )

    def check_indepentent_projections_inputs(self, key: Tensor, value: Tensor) -> None:
        k_sequence_length, k_batch_size, _ = key.shape
        v_sequence_length, v_batch_size, _ = value.shape
        is_kv_sequence_length_same = k_sequence_length == v_sequence_length
        is_kv_batch_size_same = k_batch_size == v_batch_size
        if not (is_kv_sequence_length_same and is_kv_batch_size_same):
            raise RuntimeError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )

    def check_self_attention_projection_inputs(
        self, key: Tensor, value: Tensor
    ) -> None:
        are_kv_shapes_same = key.shape == value.shape
        if not are_kv_shapes_same:
            raise RuntimeError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )

    def check_static_projection_shapes(
        self,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ):
        self.__resolve_static_projection_shape(static_keys)
        self.__resolve_static_projection_shape(static_values, True)

    def __resolve_static_projection_shape(
        self,
        static_tensor: Tensor | None = None,
        value_tensor_flag: bool = False,
    ) -> None:
        if static_tensor is not None:
            tensor_type = self.__resolve_static_projection_type(value_tensor_flag)
            expected_first_dim = self.batch_size * self.num_heads
            assert static_tensor.size(0) == expected_first_dim, (
                f"expecting {tensor_type}.size(0) of {expected_first_dim}, but got {static_tensor.size(0)}"
            )
            assert static_tensor.size(2) == self.head_dim, (
                f"expecting {tensor_type}.size(2) of {self.head_dim}, but got {static_tensor.size(2)}"
            )

    def __resolve_static_projection_type(self, value_tensor_flag: bool = False) -> str:
        return "static_values" if value_tensor_flag else "static_keys"
