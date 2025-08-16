import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, embedding

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.attention.attention import MultiHeadAttentionConfig


class AttentionUtils:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "AttentionValidator",
        key_bias_vector: Tensor | None = None,
        value_bias_vector: Tensor | None = None,
    ):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.validator = validator
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
        self.batch_first_flag = self.cfg.batch_first_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length
        self.head_dim = self.embedding_dim // self.num_heads

        self.value_bias_vector = value_bias_vector
        self.key_bias_vector = key_bias_vector

    def maybe_transpose_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        should_transpose_qkv = (
            self.batch_first_flag and self.validator.is_tensor_batched(query)
        )
        if not should_transpose_qkv:
            return query, key, value
        if key is value:
            return self.__transpose_shared_qkv(query, key)
        return (tensor.transpose(0, 1) for tensor in (query, key, value))

    def __transpose_shared_qkv(self, query: Tensor, key: Tensor):
        if query is key:
            query = key = value = query.transpose(0, 1)
            return query, key, value
        query, key = (tensor.transpose(0, 1) for tensor in (query, key))
        value = key
        return query, key, value

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

    def add_learnable_bias_vectors(
        self,
        key_projections: Tensor,
        value_projections: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.key_bias_vector is None or self.value_bias_vector is None:
            return (
                key_projections,
                value_projections,
                key_padding_mask,
                attention_mask,
            )
        repeated_key_bias = self.key_bias_vector.repeat(1, self.batch_size, 1)
        key_projections_with_bias_vector = torch.cat(
            [key_projections, repeated_key_bias]
        )
        repeated_value_bias = self.value_bias_vector.repeat(1, self.batch_size, 1)
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
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validator.check_static_projection_shapes(static_keys, static_values)

        query = self.__reshape_projection_tesnor(query)
        key = self.__reshape_projection_tesnor(key, static_keys)
        value = self.__reshape_projection_tesnor(value, static_values)

        return query, key, value

    def __reshape_projection_tesnor(
        self,
        tensor: Tensor,
        static_tensor: Tensor | None = None,
    ) -> Tensor:
        if static_tensor is not None:
            return static_tensor

        sequence_length = tensor.shape[0]
        shape = (sequence_length, self.batch_size * self.num_heads, self.head_dim)
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

    def merge_masks(
        self,
        key: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask

        source_sequence_length = key.size(1)

        shape_view = (self.batch_size, 1, 1, self.source_sequence_length)
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


class AttentionProcessor:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "AttentionValidator",
        output_model: nn.Module,
    ):
        self.cfg = cfg
        self.validator = validator
        self.num_heads = self.cfg.num_heads
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.dropout_probability = self.cfg.dropout_probability
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.average_attention_weights_flag = self.cfg.average_attention_weights_flag
        self.head_dim = self.embedding_dim // self.num_heads
        self.output_model = output_model

    def compute_attetnion(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.need_weights:
            return self.__compute_attention_with_weights(
                query, key, value, attention_mask
            )
        return self.__default_case(query, key, value, attention_mask)

    def __compute_attention_with_weights(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        weights = self.__compute_masked_attention_weights(query, key, attention_mask)
        weighted_value = self.__compute_weighted_values(weights, value)
        output = self.__compute_attention_output(weighted_value)
        output, weights = self.__prepare_output(output, weights)

        return output, weights

    def __compute_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        scaled_query = self.__scale_query(query)
        raw_weights = self.__compute_raw_masked_attention_weights(
            scaled_query, key, attention_mask
        )
        weights = F.softmax(raw_weights, dim=-1)
        if self.dropout_probability > 0.0:
            weights = F.dropout(weights, p=self.dropout_probability)
        return weights

    def __scale_query(self, query: Tensor) -> Tensor:
        head_dim = query.size(-1)
        query_scalar = math.sqrt(1.0 / float(head_dim))
        return query * query_scalar

    def __compute_raw_masked_attention_weights(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        key = key.transpose(-2, -1)
        if attention_mask is not None:
            return torch.baddbmm(attention_mask, query, key)
        return torch.bmm(query, key)

    def __compute_weighted_values(
        self,
        attention_weights: Tensor,
        values: Tensor,
    ) -> Tensor:
        assert self.target_sequence_length == self.source_sequence_length, (
            f"At the moment different source and target sequence lengths are not supported so `target_sequence_length`: {self.target_sequence_length} must be equal to `source_sequence_length`:{self.source_sequence_length}. See if this can be fixed in the future."
        )
        # The reason this does not work is because `weighted_values` after torch.bmm(attention_weights, values)
        # need to be reshaped into `weighted_values_output_shape`, if `source_sequence_length` and
        # `target_sequence_length` are different then the reshaping will not work.
        weighted_values_output_shape = (
            self.target_sequence_length * self.batch_size,
            self.embedding_dim,
        )
        weighted_values = torch.bmm(attention_weights, values)
        weighted_values = weighted_values.transpose(0, 1)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(weighted_values_output_shape)

    def __compute_attention_output(self, weighted_value: Tensor) -> Tensor:
        attention_output = self.output_model(weighted_value)
        embedding_dim = attention_output.size(-1)
        return attention_output.view(
            self.target_sequence_length,
            self.batch_size,
            embedding_dim,
        )

    def __prepare_output(
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
        attention_weights = self.__maybe_average_attention_weights(attention_weights)

        return self.__handle_batched_input(attention_output, attention_weights)

    def __maybe_average_attention_weights(self, attention_weights: Tensor) -> Tensor:
        if self.average_attention_weights_flag:
            return attention_weights.mean(dim=1)
        return attention_weights

    def __handle_batched_input(
        self, attention_output: Tensor, attention_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not self.validator.get_batched_input_flag():
            output_with_removed_batch_dim = attention_output.squeeze(1)
            weights_with_removed_batch_dim = attention_weights.squeeze(0)
            return output_with_removed_batch_dim, weights_with_removed_batch_dim
        return attention_output, attention_weights

    def __default_case(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, None]:
        attention_mask = self.__prepare_attnetion_mask(attention_mask)
        query, key, value = self.__reshape_qkv_for_attention(query, key, value)
        weighted_values = self.__compute_weighted_values_default(
            query,
            key,
            value,
            attention_mask,
            is_causal,
        )
        attention_output = self.__compute_attention_output(weighted_values)

        if not is_batched:
            return attention_output.squeeze(1), None
        return attention_output, None

    def __prepare_attnetion_mask(
        self, attention_mask: Tensor | None = None
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        is_mask_single_batch = attention_mask.size(0) == 1
        is_mask_batched = attention_mask.dim() == 3
        if is_mask_single_batch and is_mask_batched:
            return attention_mask.unsqueeze(0)
        return attention_mask.view(
            self.batch_size,
            self.num_heads,
            -1,
            self.source_sequence_length,
        )

    def __reshape_qkv_for_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        q_shape = (
            self.batch_size,
            self.num_heads,
            self.target_sequence_length,
            self.head_dim,
        )
        kv_shape = (
            self.batch_size,
            self.num_heads,
            self.source_sequence_length,
            self.head_dim,
        )
        query = query.view(q_shape)
        key = key.view(kv_shape)
        value = value.view(kv_shape)
        return query, key, value

    def __compute_weighted_values_default(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        is_causal: bool = False,
    ):
        weighted_values = F.scaled_dot_product_attention(
            query, key, value, attention_mask, self.dropout_probability, is_causal
        )

        weighted_values = weighted_values.permute(2, 0, 1, 3)
        weighted_values = weighted_values.contiguous()
        return weighted_values.view(
            self.batch_size * self.target_sequence_length,
            self.embedding_dim,
        )


class AttentionProcessor:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "AttentionValidator",
        output_model: nn.Module,
    ):
        self.cfg = cfg
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag

        if self.return_attention_weights_flag:
            self.processor = AttentionProcessorWithReturnedWeights(
                cfg, validator, output_model
            )
        else:
            self.processor = AttentionProcessorDefault(cfg, validator, output_model)

    def compute_attetnion(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.processor.compute_attention(query, key, value, attention_mask)


class AttentionProjector:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "AttentionValidator",
        qkv_model: nn.Module | None = None,
        query_model: nn.Module | None = None,
        key_model: nn.Module | None = None,
        value_model: nn.Module | None = None,
    ):
        self.cfg = cfg
        self.validator = validator

        self.query_model = query_model
        self.key_model = key_model
        self.value_model = value_model
        self.qkv_model = qkv_model

    def compute_qkv_projections(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        are_qkv_same = key is value and query is key
        if are_qkv_same:
            self.validator.check_self_attention_projection_inputs(key, value)
            return self.__compute_self_attention_projections(query)
        self.validator.check_indepentent_projections_inputs(key, value)
        return self.__compute_indepentet_projections(query, key, value)

    def __compute_self_attention_projections(
        self, query: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        qkv_projection = self.__compute_projection(query, self.qkv_model)
        query_projections, key_projections, value_projections = (
            self.__split_self_attention_projection(qkv_projection)
        )
        return query_projections, key_projections, value_projections

    def __split_self_attention_projection(
        self, qkv_projections: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        projections = qkv_projections.unflatten(-1, (3, -1))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query_projections = projections[0]
        key_projections = projections[1]
        value_projections = projections[2]

        return query_projections, key_projections, value_projections

    def __compute_indepentet_projections(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_projections = self.__compute_projection(query, self.query_model)
        key_projections = self.__compute_projection(key, self.key_model)
        value_projections = self.__compute_projection(value, self.value_model)

        return query_projections, key_projections, value_projections

    def __compute_projection(self, tensor: Tensor, model: nn.Module) -> Tensor:
        sequence_length, batch_size, embedding_dim = tensor.shape
        tensor_reshaped = tensor.view(-1, embedding_dim)
        projections = model(tensor_reshaped)
        if isinstance(projections, tuple):
            projections, _ = projections
        return projections.view(sequence_length, batch_size, -1)


class AttentionMask:
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig",
        validator: "AttentionValidator",
    ):
        self.cfg = cfg
        self.validator = validator
        self.target_dtype = self.cfg.target_dtype
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag

    def validate_padding_and_attention_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
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
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        return key_padding_mask, attention_mask

    def __validate_attention_mask(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        need_weights: bool = False,
    ) -> Tensor | None:
        if (
            self.causal_attention_mask_flag
            and key_padding_mask is None
            and not need_weights
        ):
            return

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

        self.validator.is_mask_float_or_bool(mask, mask_name)
        self.validator.is_mask_correct_dtype(
            mask, mask_name, other_type, other_name, check_other
        )

        if not torch.is_floating_point(mask):
            mask_placeholder = torch.zeros_like(mask, dtype=target_type)
            mask = mask_placeholder.masked_fill_(mask, float("-inf"))
        return mask


class AttentionValidator:
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
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_sequence_length = self.cfg.target_sequence_length

    def assert_correct_head_dim(self, head_dim: int) -> None:
        assert (head_dim * self.num_heads) == self.embeding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def is_mask_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        is_float_float = torch.is_floating_point(mask)
        is_mask_bool = mask.dtype == torch.bool
        is_mask_float_or_bool = not is_mask_bool and not is_float_float
        if is_mask_float_or_bool:
            raise RuntimeError(
                f"only bool and floating types of {mask_name} are supported"
            )

    def is_mask_correct_dtype(
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
                raise RuntimeError(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )

    def multi_head_attention_input_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> bool:
        self.batched_input_flag = self.is_tensor_batched(query)

        self.__check_query_dims(query)
        self.__check_query_key_value_dimensions(key, value)
        self.__check_key_padding_mask_dimensions(key_padding_mask)
        self.__check_attention_mask(attention_mask)
        self.__ensure_attention_mask_if_causal(attention_mask)

        return self.batched_input_flag

    def __check_query_dims(self, query: Tensor) -> None:
        if query.dim() not in (2, 3):
            raise RuntimeError(
                f"Query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D tensor"
            )

    def __check_query_key_value_dimensions(self, key: Tensor, value: Tensor) -> None:
        expected_dims = 3 if self.batched_input_flag else 2
        qk_dimension_check = key.dim() == expected_dims
        qv_dimension_check = value.dim() == expected_dims
        are_qkv_dimensions_same = qk_dimension_check and qv_dimension_check
        if not are_qkv_dimensions_same:
            raise RuntimeError(
                f"For {self.__format_dimension_context()} query, expected key and value to be {expected_dims}-D "
                f"but found {key.dim()}-D and {value.dim()}-D tensors respectively"
            )

    def __check_key_padding_mask_dimensions(
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

    def __check_attention_mask(
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

        expected_shape = self.__resolve_attention_mask_shape(attention_mask)
        if attention_mask.shape != expected_shape:
            raise RuntimeError(
                f"Expected `attention_mask` shape to be {expected_shape} but got {attention_mask.shape}"
            )

    def __format_dimension_context(self) -> str:
        return "batched (3-D)" if self.batched_input_flag else "unbatched (2-D)"

    def __resolve_attention_mask_shape(
        self, attention_mask: Tensor
    ) -> tuple[int, int, int] | tuple[int, int]:
        if attention_mask.dim() == 3:
            return (
                self.batch_size * self.num_heads,
                self.source_sequence_length,
                self.target_sequence_length,
            )
        return (self.source_sequence_length, self.target_sequence_length)

    def __ensure_attention_mask_if_causal(
        self,
        attention_mask: Tensor | None = None,
    ) -> None:
        if self.causal_attention_mask_flag and attention_mask is None:
            raise RuntimeError(
                "Need `attention_mask` if specifying the `causal_attention_mask_flag` hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

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
