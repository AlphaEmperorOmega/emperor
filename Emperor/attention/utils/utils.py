import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module, device

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class AttentionUtils:
    def __init__(self, validator: "AttentionValidator"):
        self.validator = validator
        self.batch_first_flag = None

    def transpose_qkv_if_batched(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        batched_input_flag = query.dim() == 3
        is_batch_first_flag = self.batch_first_flag and batched_input_flag
        if not is_batch_first_flag:
            return query, key, value

        if key is value:
            query, key, value = self.__transpose_shared_qkv(query, key)
            return query, key, value

        query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        return query, key, value

    def __transpose_shared_qkv(self, query: Tensor, key: Tensor):
        if query is key:
            query = key = value = query.transpose(0, 1)
            return query, key, value
        query, key = (x.transpose(0, 1) for x in (query, key))
        value = key
        return query, key, value

    def add_batch_dimension_if_missing(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if self.validator.is_input_batched(query):
            return query, key, value, key_padding_mask
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        return query, key, value, key_padding_mask

    def add_bias_vectors_to_kv(
        self,
        key_projections: Tensor,
        value_projections: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.bias_key is None or self.bias_value is None:
            return (
                key_projections,
                value_projections,
                key_padding_mask,
                attention_mask,
            )
        repeated_key_bias = self.key_bias.repeat(1, self.batch_size, 1)
        repeated_value_bias = self.value_bias.repeat(1, self.batch_size, 1)
        key_projections_with_bias_vector = torch.cat(
            [key_projections, repeated_key_bias]
        )
        value_projections_with_bias_vector = torch.cat(
            [value_projections, repeated_value_bias]
        )
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        return (
            key_projections_with_bias_vector,
            value_projections_with_bias_vector,
            key_padding_mask,
            attention_mask,
        )

    def prepare_qkv_projection_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ):
        self.assert_correct_static_projection_shapes()
        query_shape = self.__get_expected_qkv_shapes(self.target_sequence_length)
        query = query.view(query_shape)
        query = query.transpose(0, 1)

        updated_keys = static_keys
        if static_keys is None:
            key_sequence_length = key.shape[0]
            key_shape = self.__get_expected_qkv_shapes(key_sequence_length)
            updated_keys = updated_keys.view(key_shape)
            updated_keys = updated_keys.transpose(0, 1)

        updated_values = static_values
        if static_values is None:
            value_sequence_length = value.shape[0]
            value_shape = self.__get_expected_qkv_shapes(value_sequence_length)
            updated_values = updated_values.view(value_shape)
            updated_values = updated_values.transpose(0, 1)

        return query, key, value

    def __get_expected_qkv_shapes(self, sequence_length):
        return sequence_length, bsz * num_heads, head_dim


class AttentionProcessor:
    def __init__(self):
        self.attention_output_model = nn.Linear(1, 1)
        # self.need_weights = self.cfg.need_weights

    def compute_attetnion(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.need_weights:
            return self.__need_weights_case(query, key, value, attention_mask)
        return self.__default_case(query, key, value, attention_mask)

    def __need_weights_case(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        attention_weights = self.__compute_attention_masked(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(attention_weights, value)
        attention_output = self.__compute_attention_output(weighted_value)
        attention_output, attention_weights = self.__prepare_attention_output(
            attention_output, attention_weights
        )

        return attention_output, attention_weights

    def __prepare_attention_output(
        self,
        attention_output: Tensor,
        attention_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        attention_weights_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.source_sequence_length,
        )
        attention_weights = attention_weights.view(attention_weights_shape)
        if self.average_attn_weights:
            attention_weights = attention_weights.mean(dim=1)

        if not self.is_batched:
            attention_output = attention_output.squeeze(1)
            attention_weights = attention_weights.squeeze(0)
        return attention_output, attention_weights

    def __compute_attention_masked(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        _, _, embedding_dim = query.shape
        scaled_qeury = query * math.sqrt(1.0 / float(embedding_dim))
        if attention_mask is not None:
            attention_output_weights = torch.baddbmm(
                attention_mask, scaled_qeury, key.transpose(-2, -1)
            )
            attention_output_weights = softmax(attention_output_weights, dim=-1)
        attention_output_weights = torch.bmm(scaled_qeury, key.transpose(-2, -1))
        attention_output_weights = softmax(attention_output_weights, dim=-1)
        return attention_output_weights

    def __compute_weighted_values(self, attention_weights: Tensor, values: Tensor):
        weighted_values_shape = (
            self.target_sequence_length * self.batch_size,
            self.embedding_dim,
        )
        weighted_values = torch.bmm(attention_weights, values)
        weighted_values = weighted_values.transpose(0, 1)
        weighted_values = weighted_values.contiguous()
        weighted_values = weighted_values.view(weighted_values_shape)
        return weighted_values

    def __compute_attention_output(self, weighted_value: Tensor):
        attention_output = self.attention_output_model(weighted_value)
        attention_output_shape = (
            self.target_sequence_length,
            self.batch_size,
            attention_output.size(1),
        )
        attention_output = attention_output.view(attention_output_shape)
        return attention_output

    def __default_case(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, None]:
        attention_mask = self.__prepare_attnetion_mask(attention_mask)
        query, key, value = self.__prepare_qkv_for_attention(query, key, value)
        weighted_values = self.__compute_weighted_values(
            query,
            key,
            value,
            attention_mask,
            dropout_probability,
            is_causal,
        )
        attention_output = self.__compute_attention_output(weighted_values)

        if not is_batched:
            return attention_output.squeeze(1), None
        return attention_output, None

    def __prepare_attnetion_mask(self, attention_mask: Tensor | None) -> Tensor | None:
        if attention_mask is None:
            return None
        is_attention_one_batch = attention_mask.size(0) == 1
        is_attention_mask_3D = attention_mask.dim() == 3
        if is_attention_one_batch and is_attention_mask_3D:
            return attention_mask.unsqueeze(0)
        attention_mask_shape = (
            self.batch_size,
            self.num_heads,
            -1,
            self.source_sequence_length,
        )
        attention_mask = attention_mask.view(attention_mask_shape)
        return attention_mask

    def __prepare_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        qery_shape = (self.bsz, self.num_heads, self.tgt_len, self.head_dim)
        key_value_shape = (self.bsz, self.num_heads, self.src_len, self.head_dim)
        query = query.view(qery_shape)
        key = key.view(key_value_shape)
        value = value.view(key_value_shape)
        return query, key, value

    def __compute_weighted_values(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        dropout_probability: float,
        is_causal: bool = False,
    ):
        weighted_values = scaled_dot_product_attention(
            query, key, value, attention_mask, dropout_probability, is_causal
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        weighted_values_shape = (
            self.batch_size * self.target_sequence_length,
            self.embedding_dim,
        )
        weighted_values = weighted_values.view(weighted_values_shape)

        return weighted_values


class AttentionProjector:
    def __init__(self, validator: "AttentionValidator"):
        self.validator = validator

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        indepentent_projection_flag: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.assert_qkv_based_on_weight_projection(
            indepentent_projection_flag
        )
        are_qkv_same = key is value and query is key
        if are_qkv_same:
            return self.__compute_self_projections(query)
        return self.__compute_indepentet_projections(query, key, value)

    def __compute_self_projections(
        self, query: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        shared_qkv_projection = self.shared_projection_model(query)
        projections = shared_qkv_projection.unflatten(-1, (3, self.embedding_dim))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query_projections, key_projections, value_projections = (
            projections[0],
            projections[1],
            projections[2],
        )
        return query_projections, key_projections, value_projections

    def __compute_indepentet_projections(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_projections = self.query_module(query)
        key_projections = self.key_module(key)
        value_projections = self.value_module(value)
        return query_projections, key_projections, value_projections


class AttentionMask:
    def __init__(
        self,
        validator: "AttentionValidator",
        target_dtype: "DType",
        causal_attention_mask_flag: bool,
    ):
        self.validator = validator
        self.target_dtype = target_dtype
        self.causal_attention_mask_flag = causal_attention_mask_flag

    def merge_masks(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _check_key_padding_mask(key_padding_mask, src_len, bsz)

        key_padding_mask_shape = (self.batch_size, 1, 1, self.source_sequence_length)
        key_padding_mask = (
            key_padding_mask.view(key_padding_mask_shape)
            .expand(-1, self.num_heads, -1, -1)
            .reshape(self.batch_size * self.num_heads, 1, self.source_sequence_length)
        )
        attention_mask = key_padding_mask
        if attention_mask is not None:
            attention_mask = attention_mask + key_padding_mask

        return attention_mask

    def validate_padding_and_attention_masks(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        need_weights: bool = False,
    ) -> tuple[Tensor | None, Tensor | None]:
        key_padding_mask = self.__canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attention_mask),
            other_name="attention_mask",
            target_type=self.target_dtype,
        )
        attention_mask = self.__validate_attention_mask(
            attention_mask,
            key_padding_mask,
            need_weights,
        )
        return key_padding_mask, attention_mask

    def __validate_attention_mask(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        need_weights: bool,
    ) -> Tensor | None:
        if (
            self.causal_attention_mask_flag
            and key_padding_mask is None
            and not need_weights
        ):
            return None

        if key_padding_mask is not None:
            self.causal_attention_mask_flag = False

        attention_mask = self.__canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=self.target_dtype,
            check_other=False,
        )

        attention_mask = self.__ensure_correct_shape(attention_mask)

        return attention_mask

    def __ensure_correct_shape(self, attention_mask: Tensor | None) -> Tensor | None:
        self.validator.assert_attention_mask_shape(attention_mask)
        if attention_mask.dim() == 2:
            return attention_mask.unsqueeze(0)
        elif attention_mask.dim() == 3:
            return attention_mask.unsqueeze(1)
        return attention_mask

    def __canonical_mask(
        self,
        mask: Tensor | None,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        target_type: "DType",
        check_other: bool = True,
    ) -> Tensor | None:
        if mask is None:
            return mask

        self.validator.assert_mask_float_or_bool(mask, mask_name)
        self.validator.assert_correct_mask_dtype(
            mask, mask_name, other_type, other_name, check_other
        )

        if not torch.is_floating_point(mask):
            mask = torch.zeros_like(mask, dtype=target_type)
            mask = mask.masked_fill(mask, float("-inf"))
        return mask

    def get_causal_attention_mask_flag(self) -> bool:
        return self.causal_attention_mask_flag


class AttentionValidator:
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        causal_attention_mask_flag: bool = False,
    ):
        self.num_heads = num_heads
        self.config_embeding_dim = embedding_dim
        self.causal_attention_mask_flag = causal_attention_mask_flag
        self.query_dims = None
        self.key_dims = None
        self.value_dims = None
        self.embedding_dim = None
        self.key_sequence_length = None
        self.batched_input_flag = None
        self.query_sequence_length = None

    def assert_correct_head_dim(self, head_dim: int) -> None:
        assert (head_dim * self.num_heads) == self.config_embeding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def assert_mask_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        is_float_float = torch.is_floating_point(mask)
        is_mask_bool = mask.dtype == torch.bool
        is_mask_float_or_bool = not is_mask_bool and not is_float_float
        if is_mask_float_or_bool:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )

    def assert_correct_mask_dtype(
        self,
        mask: Tensor,
        mask_name: str,
        other_type: "DType | None",
        other_name: str,
        check_other: bool = True,
    ):
        mask_dtype = mask.dtype
        should_check_other_dtype = check_other and other_type is not None
        if should_check_other_dtype:
            does_dtype_match = mask_dtype == other_type
            if not does_dtype_match:
                raise AssertionError(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )

    def assert_multi_head_attention_shape(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> bool:
        self.batched_input_flag = query.dim() == 3
        self.query_dims = query.dim()
        self.key_dims = key.dim()
        self.value_dims = value.dim()
        self.embedding_dim = query.shape[-1]
        self.query_sequence_length = query.shape[0]
        self.key_sequence_length = value.shape[0]

        self.__check_query_dims()
        self.__check_query_key_value_dimensions()
        self.__check_key_padding_mask_dimensions(key_padding_mask)
        self.__check_attention_mask_dimensions(attention_mask)
        self.__ensure_attention_mask_if_causal(attention_mask)

        return self.batched_input_flag

    def __check_query_dims(self) -> None:
        if self.query_dims not in (2, 3):
            raise AssertionError(
                f"Query should be unbatched 2D or batched 3D tensor but received {self.query_dims}-D tensor"
            )

    def __check_query_key_value_dimensions(self) -> None:
        expected_dims = 3 if self.batched_input_flag else 2
        qk_dimension_check = self.key_dims == expected_dims
        qv_dimension_check = self.value_dims == expected_dims
        are_qkv_dimensions_same = qk_dimension_check and qv_dimension_check
        assert are_qkv_dimensions_same, (
            f"For {self.__format_dimension_context()} query, expected key and value to be {expected_dims}-D "
            f"but found {self.key_dims}-D and {self.value_dims}-D tensors respectively"
        )

    def __check_key_padding_mask_dimensions(
        self,
        key_padding_mask: Tensor | None = None,
    ) -> None:
        if key_padding_mask is None:
            return

        expected_dim = 2 if self.batched_input_flag else 1
        assert key_padding_mask.dim() == expected_dim, (
            f"For {self.__format_dimension_context()} query, expected `key_padding_mask` to be None or {expected_dim}-D "
            f"but found {key_padding_mask.dim()}-D tensor instead"
        )

    def __check_attention_mask_dimensions(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if attention_mask is None:
            return

        attention_mask_dimensions = attention_mask.dim()
        assert attention_mask_dimensions in (2, 3), (
            f"For {self.__format_dimension_context()} query, expected attention_mask to be None, 2-D, or 3-D "
            f"but found {attention_mask_dimensions}-D tensor instead"
        )

        if attention_mask_dimensions == 3:
            expected_shape = (
                self.num_heads,
                self.query_sequence_length,
                self.key_sequence_length,
            )
            assert attention_mask.shape == expected_shape, (
                f"Expected `attention_mask` shape to be {expected_shape} but got {attention_mask.shape}"
            )

    def __format_dimension_context(self) -> str:
        return "batched (3-D)" if self.has_batch_dimension else "unbatched (2-D)"

    def __ensure_attention_mask_if_causal(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if attention_mask is None:
            return

        ensure_attention_mask_if_causal = (
            self.causal_attention_mask_flag and attention_mask is None
        )
        if ensure_attention_mask_if_causal:
            raise RuntimeError(
                "Need `attention_mask` if specifying the `causal_attention_mask_flag` hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

    def is_input_batched(self, query: Tensor | None = None) -> bool:
        if self.has_batch_dimension is None:
            assert query is not None, (
                "Query tensor must be provided to check batch dimension."
            )
            return query.dim() == 3
        return self.has_batch_dimension

    def assert_correct_embedding_dim(self, expected_embedding_dim: int):
        assert self.embedding_dim == expected_embedding_dim, (
            f"Was expecting embedding dimension of {expected_embedding_dim}, but got {self.embedding_dim}"
        )

    def assert_separate_projection_layer(self):
        assert key.shape == value.shape, (
            f"key shape {key.shape} does not match value shape {value.shape}"
        )
        assert q_proj_weight is not None, (
            "use_separate_proj_weight is True but q_proj_weight is None"
        )
        assert k_proj_weight is not None, (
            "use_separate_proj_weight is True but k_proj_weight is None"
        )
        assert v_proj_weight is not None, (
            "use_separate_proj_weight is True but v_proj_weight is None"
        )

    def assert_shared_projection_layer(self):
        assert self.key.shape[:2] == value.shape[:2], (
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        )
        assert in_proj_weight is not None, (
            "use_separate_proj_weight is False but in_proj_weight is None"
        )

    def assert_qkv_based_on_weight_projection(
        self, use_separate_projection_weight: bool
    ):
        if use_separate_projection_weight:
            self.assert_separate_projection_layer()
        else:
            self.assert_shared_projection_layer()

    def assert_attention_mask_shape(self, attention_mask):
        is_2d_mask = attention_mask.dim() == 2
        is_3d_mask = attention_mask.dim() == 3
        if is_2d_mask:
            expected_2d_mask_shape = (
                self.target_sequence_length,
                self.source_sequence_length,
            )
            if attention_mask.shape != expected_2d_mask_shape:
                raise RuntimeError(
                    f"The shape of the 2D attention_mask is {attention_mask.shape}, but should be {correct_2d_size}."
                )
        elif is_3d_mask:
            expected_3d_mask_shape = (
                self.batch_size * self.num_heads,
                self.target_sequence_length,
                self.source_sequence_length,
            )
            if expected_3d_mask_shape == 3:
                raise RuntimeError(
                    f"Expected attention_mask to be 2D or 3D, but got {attention_mask.dim()}D."
                )
        else:
            raise RuntimeError(
                f"attention_mask's dimension {attention_mask.dim()} is not supported"
            )

    def assert_correct_static_projection_shapes(
        self,
        static_keys: Tensor | None,
        static_values: Tensor | None,
    ):
        if static_keys:
            assert static_keys.size(0) == self.batch_size * self.num_heads, (
                f"expecting static_k.size(0) of {self.batch_size * self.num_heads}, but got {static_keys.size(0)}"
            )
            assert static_keys.size(2) == self.head_dim, (
                f"expecting static_k.size(2) of {self.head_dim}, but got {static_keys.size(2)}"
            )
        if static_values is not None:
            assert static_values.size(0) == self.batch_size * self.num_heads, (
                f"expecting static_v.size(0) of {self.batch_size * self.num_heads}, but got {static_values.size(0)}"
            )
            assert static_values.size(2) == self.head_dim, (
                f"expecting static_v.size(2) of {self.head_dim}, but got {static_values.size(2)}"
            )
