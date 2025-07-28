import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, embedding
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase, Module, device

from Emperor.layers.utils.enums import (
    LayerTypes,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
    from Emperor.config import ModelConfig


@dataclass
class AttentionConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={
            "help": "Type of layer used for to generate query, key, value projections."
        },
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    num_heads: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    dropout_probability: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    key_value_bias_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    zero_attention_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    key_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    value_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    batch_first_flag: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    dtype: torch.dtype | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )


class Attention(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(config, overrides)

        self.model_type = self.cfg.model_type
        self.embedding_dim = self.cfg.embedding_dim
        self.num_heads = self.cfg.num_heads
        self.dropout_probability = self.cfg.dropout_probability
        self.key_value_bias_flag = self.cfg.key_value_bias_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.batch_first_flag = self.cfg.batch_first_flag
        self.model_type = self.cfg.model_type
        self.dtype = self.cfg.dtype
        self.key_dim = self.cfg.key_dim
        self.value_dim = self.cfg.value_dim
        self.query_dim = self.embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads

        temp = nn.MultiheadAttention

        self.query_key_value_module = None
        self.query_module = None
        self.key_module = None
        self.value_module = None
        self.input_tesnor_3D_flag = None
        self.__create_projection_models(cfg)
        self.__assert_input_requirements()

    def __assert_input_requirements(self):
        assert (self.head_dim * self.num_heads) == self.embedding_dim, (
            "`embedding_dim` must be perfectly divisible by `number_of_heads`."
        )

    def __are_key_query_value_dims_equal(self) -> bool:
        are_keys_querys_same = self.key_dim == self.embed_dim
        are_values_querys_same = self.value_dim == self.embed_dim
        return are_keys_querys_same and are_values_querys_same

    def __create_projection_models(self, cfg: "ModelConfig") -> None:
        self.output_dim = self.model_type.value(cfg)
        if self.__are_key_query_value_dims_equal():
            self.query_key_value_module = self.model_type.value(cfg)
            return
        self.query_module = self.model_type.value(cfg)
        self.key_module = self.model_type.value(cfg)
        self.value_module = self.model_type.value(cfg)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attention_mask: Tensor | None = None,
        average_attention_weights: bool = False,
        causal_attention_mask: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        self.set_input_tensor_3D_flag(query)
        key_padding_mask, attention_mask = self.__update_masks(
            key_padding_mask, attention_mask, query.dtype
        )
        query, key, value = self.__resolve_query_key_value_shapes(query, key, value)

        return ()

    def is_input_tensor_3D(self) -> bool:
        if self.input_tesnor_3D_flag is None:
            AssertionError("`input_tesnor_3D_flag` flag is not set.")
        return self.input_tesnor_3D_flag

    def set_input_tensor_3D_flag(self, query: Tensor) -> None:
        self.input_tesnor_3D_flag = query.dim() == 3

    def __update_masks(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        target_type: DType,
    ) -> tuple[Tensor | None, Tensor | None]:
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attention_mask),
            other_name="attention_mask",
            target_type=target_type,
        )
        attention_mask = F._canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=target_type,
            check_other=False,
        )
        return key_padding_mask, attention_mask

    def __resolve_query_key_value_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        if self.batch_first_flag and self.is_input_tensor_3D():
            if key is value:
                if query is key:
                    key = query = value = query.transpose(0, 1)
                else:
                    query, key = (x.transpose(0, 1) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        return query, key, value


@dataclass
class MultiHeadAttentionConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={
            "help": "Type of layer used for to generate query, key, value projections."
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    target_dtype: DType | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )
    use_separate_projection_weight: bool | None = field(
        default=None,
        metadata={"help": "Expert input dimension"},
    )


class MultiHeadAttentionBehaviour(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(config, overrides)

        self.model_type = self.cfg.model_type
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.target_dtype = self.cfg.target_dtype
        self.use_separate_projection_weight = self.cfg.use_separate_projection_weight
        self._valudate_fields(self.cfg, MultiHeadAttentionConfig)

        self.validator = AttentionValidator(self.num_heads)
        self.masks = AttentionMaskUpdates()
        self.ptrojections = AttentionProjections()

        self.query_model = self.model_type.value(cfg)
        self.key_model = self.model_type.value(cfg)
        self.value_model = self.model_type.value(cfg)

        self.shared_projection_model = self.model_type.value(cfg)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        input_projection_weight: Tensor | None,
        input_projection_bias: Tensor | None,
        bias_key: Tensor | None,
        bias_value: Tensor | None,
        add_zero_attention: bool,
        dropout_probability: float,
        output_projection_weight: Tensor,
        output_projection_bias: Tensor | None,
        training: bool = True,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attention_mask: Tensor | None = None,
        use_separate_projection_weight: bool = False,
        query_projection_weight: Tensor | None = None,
        key_projection_weight: Tensor | None = None,
        value_projection_weight: Tensor | None = None,
        static_key: Tensor | None = None,
        static_values: Tensor | None = None,
        average_attention_weights: bool = True,
        is_causal: bool = False,
    ):
        self.target_dtype = query.dtype
        self.validator.assert_shapes(
            query, key, value, key_padding_mask, attention_mask
        )
        query, key, value, key_padding_mask = self.__add_batch_dimension_if_missing(
            query, key, value, key_padding_mask
        )

        key_padding_mask, attention_mask, causal_attention_mask_flag = (
            self.masks.create_masks(
                attention_mask,
                key_padding_mask,
                causal_attention_mask_flag,
                need_weights,
            )
        )

        head_dim = self.__resolve_head_dim(self.embedding_dim)
        query_projections, key_projections, value_projections = (
            self.ptrojections.compute_qkv_projections(query, key, value)
        )
        attention_mask = self.__updated_attention_mask(attention_mask)
        (
            key_projections,
            value_projections,
            key_padding_mask,
            attention_mask,
        ) = self.__add_bias_vectors_to_kv(
            query,
            key,
            value,
            attention_mask,
            key_padding_mask,
        )
        reshaped_query, reshaped_key, reshaped_value = self.__reshape_qkv_projection(
            query, key, value, static_key, static_values
        )

    def __add_batch_dimension_if_missing(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if self.shape_validator.is_input_batched(query):
            return query, key, value, key_padding_mask
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        return query, key, value, key_padding_mask

    def __resolve_head_dim(self, embedding_dim: int | Tensor) -> Tensor:
        self.shape_validator.assert_correct_embedding_dim(embedding_dim)
        if isinstance(embedding_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embedding_dim.div(self.num_heads, rounding_mode="trunc")
        else:
            head_dim = embedding_dim // self.num_heads
        self.shape_validator.assert_correct_head_dim(head_dim)
        return head_dim

    def __updated_attention_mask(self, attention_mask: Tensor | None):
        self.validator.assert_attention_mask_shape(attention_mask)
        is_2d_mask = attention_mask.dim() == 2
        is_3d_mask = attention_mask.dim() == 3
        if is_2d_mask:
            attention_mask = attention_mask.unsqueeze(0)
        elif is_3d_mask:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def __add_bias_vectors_to_kv(
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

    def __reshape_qkv_projection(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        static_keys: Tensor | None = None,
        static_values: Tensor | None = None,
    ):
        self.assert_correct_static_projection_shapes()
        query = query.view(self.__get_expected_qkv_shapes(self.target_sequence_length))
        query = query.transpose(0, 1)
        if static_keys is None:
            key_sequence_length = key.shape[0]
            key = key.view(self.__get_expected_qkv_shapes(key_sequence_length))
            key = key.transpose(0, 1)
        else:
            key = static_keys
        if static_values is None:
            value_sequence_length = value.shape[0]
            value = value.view(self.__get_expected_qkv_shapes(value_sequence_length))
            value = value.transpose(0, 1)
        else:
            value = static_values

        return query, key, value

    def __get_expected_qkv_shapes(self, sequence_length):
        return sequence_length, bsz * num_heads, head_dim


class AttentionProjections:
    def __init__(self):
        self.validator = AttentionValidator()

    def compute_qkv_projections(
        self,
        use_separate_projection_weight: bool = False,
    ):
        self.validator.assert_qkv_based_on_weight_projection(
            use_separate_projection_weight
        )
        embedding_dim = query.size(-1)
        are_key_values_same = key is value
        are_query_key_same = query is key

        if use_separate_projection_weight:
            if are_key_values_same:
                if are_query_key_same:
                    return self.__compute_self_projections(query, key, value)
                else:
                    return self.__compute_ecnoder_decoder_projections(query, key, value)
        else:
            return self.__compute_projections(query, key, value)

    def __compute_self_projections(self):
        shared_qkv_projection = self.shared_projection_model(query)
        projections = shared_qkv_projection.unflatten(-1, (3, embedding_dim))
        projections = projections.unsqueeze(0)
        projections = projections.transpose(0, -2)
        projections = projections.squeeze(-2)
        projections = projections.contiguous()
        query, key, value = projections[0], projections[1], projections[2]
        return query, key, value

    def __compute_ecnoder_decoder_projections(self):
        pass

    def __compute_projections(self):
        query_projections = self.query_module(query)
        key_projections = self.key_module(key)
        value_projections = self.value_module(value)
        return query_projections, key_projections, value_projections


class AttentionMaskUpdates:
    def __init__(self, target_dtype: DType):
        self.target_dtype = target_dtype
        self.validator = AttentionValidator()

        pass

    def create_masks(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        causal_attention_mask_flag: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor | None, Tensor | None, bool]:
        key_padding_mask = self.__canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attention_mask),
            other_name="attention_mask",
            target_type=self.target_dtype,
        )
        attention_mask, causal_attention_mask_flag = self.__validate_attention_mask(
            attention_mask,
            key_padding_mask,
            causal_attention_mask_flag,
            need_weights,
        )
        return key_padding_mask, attention_mask, causal_attention_mask_flag

    def __validate_attention_mask(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        causal_attention_mask_flag: bool,
        need_weights: bool,
    ) -> tuple[Tensor | None, bool]:
        if causal_attention_mask_flag and key_padding_mask is None and not need_weights:
            return None, causal_attention_mask_flag

        if key_padding_mask is not None:
            causal_attention_mask_flag = False

        attention_mask = self.__canonical_mask(
            mask=attention_mask,
            mask_name="attention_mask",
            other_type=None,
            other_name="",
            target_type=self.target_dtype,
            check_other=False,
        )

        return attention_mask, causal_attention_mask_flag

    def __canonical_mask(
        self,
        mask: Tensor | None,
        mask_name: str,
        other_type: DType | None,
        other_name: str,
        target_type: DType,
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


class AttentionValidator:
    def __init__(
        self,
        num_heads: int,
        causal_attention_mask_flag: bool = False,
    ):
        self.num_heads = num_heads
        self.causal_attention_mask_flag = causal_attention_mask_flag
        self.query_dims = None
        self.key_dims = None
        self.value_dims = None
        self.embedding_dim = None
        self.key_sequence_length = None
        self.has_batch_dimension = None
        self.query_sequence_length = None

    def assert_mask_float_or_bool(
        self,
        mask: Tensor,
        mask_name: str,
    ) -> None:
        mask_dtype = mask.dtype
        mask_float_check = torch.is_floating_point(mask)
        is_mask_bool = mask_dtype == torch.bool
        is_mask_float_or_bool = is_mask_bool and not mask_float_check
        if is_mask_float_or_bool:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )

    def assert_correct_mask_dtype(
        self,
        mask: Tensor,
        mask_name: str,
        other_type: DType | None,
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

    def assert_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> bool:
        self.has_batch_dimension = query.dim() == 3
        self.query_dims = query.dim()
        self.key_dims = key.dim()
        self.value_dims = value.dim()
        self.embedding_dim = query.shape[-1]
        self.query_sequence_length = query.shape[0]
        self.key_sequence_length = value.shape[0]

        self.__check_query_dims()
        self.__check_key_value_dims()
        self.__check_key_padding_mask_dims(key_padding_mask)
        self.__check_attention_mask_dims(attention_mask)
        self.__ensure_attention_mask_if_causal(attention_mask)

        return self.has_batch_dimension

    def __check_query_dims(self) -> None:
        if self.query_dims not in (2, 3):
            raise AssertionError(
                f"Query should be unbatched 2D or batched 3D tensor but received {self.query_dims}-D tensor"
            )

    def __check_key_value_dims(self) -> None:
        expected_dim = 3 if self.has_batch_dimension else 2
        query_key_shape_check = self.key_dims == expected_dim
        query_value_shapes_check = self.value_dims == expected_dim
        are_qkv_shapes_same = query_key_shape_check and query_value_shapes_check
        assert are_qkv_shapes_same, (
            f"For {self.__format_dimension_context()} query, expected key and value to be {expected_dim}-D "
            f"but found {self.key_dims}-D and {self.value_dims}-D tensors respectively"
        )

    def __check_key_padding_mask_dims(
        self, key_padding_mask: Tensor | None = None
    ) -> None:
        if key_padding_mask is None:
            return

        expected_dim = 2 if self.has_batch_dimension else 1
        key_padding_dims = key_padding_mask.dim()
        key_padding_mask_dims_check = key_padding_dims == expected_dim
        assert key_padding_mask_dims_check, (
            f"For {self.__format_dimension_context()} query, expected `key_padding_mask` to be None or {expected_dim}-D "
            f"but found {key_padding_dims}-D tensor instead"
        )

    def __check_attention_mask_dims(self, attention_mask: Tensor | None = None) -> None:
        if attention_mask is None:
            return

        attention_mask_dims = attention_mask.dim()
        assert attention_mask_dims in (2, 3), (
            f"For {self.__format_dimension_context()} query, expected attn_mask to be None, 2-D, or 3-D "
            f"but found {attention_mask_dims}-D tensor instead"
        )

        if attention_mask_dims == 3:
            expected_shape = (
                self.num_heads,
                self.query_sequence_length,
                self.key_sequence_length,
            )
            attention_mask_shape_check = attention_mask.shape == expected_shape
            assert attention_mask_shape_check, (
                f"Expected `attention_mask` shape to be {expected_shape} but got {attention_mask.shape}"
            )

    def __format_dimension_context(self) -> str:
        return "batched (3-D)" if self.has_batch_dimension else "unbatched (2-D)"

    def __ensure_attention_mask_if_causal(
        self, attention_mask: Tensor | None = None
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

    def assert_correct_head_dim(self, head_dim: int):
        assert head_dim * self.num_heads == self.embedding_dim, (
            f"`embed_dim` {self.embedding_dim} not divisible by num_heads {self.num_heads}"
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
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
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
