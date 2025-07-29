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
        self.attention_computation = AttetentionComputation()

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
        query, key, value = self.__prepare_qkv_projection_for_attention(
            query, key, value, static_key, static_values
        )
        key, value, attention_mask, key_padding_mask = self.__add_zero_attention(
            key, value, attention_mask, key_padding_mask
        )
        updated_source_sequence_length = key.size(1)
        attention_mask, key_padding_mask = self.__update_key_padding_mask(
            attention_mask, key_padding_mask
        )
        attention_output, attention_weights = (
            self.attention_computation.compute_attetnion()
        )
        return attention_output, attention_weights

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

    def __prepare_qkv_projection_for_attention(
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

    def __add_zero_attention(
        self,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if not self.zero_attention_flag:
            return key, value, attention_mask, key_padding_mask

        zero_attention_shape = (self.batch_size * self.num_heads, 1, self.head_dim)
        key_zeros = torch.zeros(
            zero_attention_shape, dtype=key.dtype, device=key.device
        )
        key = torch.cat([key, key_zeros], dim=1)
        value_zeros = torch.zeros(
            zero_attention_shape, dtype=value.dtype, device=value.device
        )
        value = torch.cat([value, value_zeros], dim=1)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

        return key, value, attention_mask, key_padding_mask

    def __update_masks(
        self,
        attention_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return attention_mask, key_padding_mask

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

        return attention_mask, key_padding_mask


class AttetentionComputation:
    def __init__(self):
        self.attention_output_model = nn.Linear()
        self.need_weights = self.cfg.need_weights

    def compute_attetnion(self):
        if self.need_weights:
            attention_output, attention_output_weights = self.__need_weights_case()
            return attention_output, attention_output_weights
        else:
            output = self.__default_case()
            return output, None

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

    def __default_case(self):
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
            f"For {self.__format_dimension_context()} query, expected attention_mask to be None, 2-D, or 3-D "
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


