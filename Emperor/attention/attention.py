import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from Emperor.attention.utils.utils import (
    AttentionMask,
    AttentionProcessor,
    AttentionProjector,
    AttentionUtils,
    AttentionValidator,
)
from Emperor.base.utils import DataClassBase, Module, device

from Emperor.layers.utils.enums import (
    LayerTypes,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
    from Emperor.config import ModelConfig


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
        metadata={"help": ""},
    )
    num_heads: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    key_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    value_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    target_dtype: "DType | None" = field(
        default=None,
        metadata={"help": ""},
    )
    use_separate_projection_weight: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    key_value_bias_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    zero_attention_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    batch_first_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )


class MultiHeadAttention(Module):
    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "MultiHeadAttentionConfig" = self._overwrite_config(config, overrides)

        self.batch_size = self.cfg.batch_size
        self.model_type = self.cfg.model_type
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
        self.query_dim = self.embedding_dim
        self.key_dim = self.cfg.key_dim
        self.value_dim = self.cfg.value_dim
        self.target_dtype = self.cfg.target_dtype
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.use_separate_projection_weight = self.cfg.use_separate_projection_weight
        self.dropout_probability = self.cfg.dropout_probability
        self.key_value_bias_flag = self.cfg.key_value_bias_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.batch_first_flag = self.cfg.batch_first_flag
        self.model_type = self.cfg.model_type
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.__resolve_kv_dimensions()
        self._valudate_fields(self.cfg, MultiHeadAttentionConfig)

        m = self.__build_projection_models(cfg)
        if len(m) == 4:
            self.query_model, self.key_model, self.value_model, self.output_model = m
        else:
            self.qkv_model, self.output_model = m

        self.__initialize_attention_components()
        self.head_dim = self.__resolve_head_dim()

    def __resolve_head_dim(self):
        head_dim = self.embedding_dim // self.num_heads
        self.validator.assert_correct_head_dim(head_dim)
        return head_dim

    def __resolve_kv_dimensions(self):
        self.key_dim = self.embedding_dim if self.cfg.key_dim == 0 else self.cfg.key_dim
        self.value_dim = (
            self.embedding_dim if self.cfg.value_dim == 0 else self.cfg.value_dim
        )

    def __initialize_attention_components(self):
        self.validator = AttentionValidator(self.cfg)
        self.masks = AttentionMask(self.cfg, self.validator)
        self.projector = AttentionProjector(
            self.cfg,
            self.validator,
            self.qkv_model,
            self.query_model,
            self.key_model,
            self.value_model,
        )
        self.processor = AttentionProcessor(self.cfg)
        self.utils = AttentionUtils(self.cfg, self.validator)

    def __build_projection_models(self, cfg: "ModelConfig") -> tuple:
        if self.__are_qkv_dimensions_equal():
            return self.__build_shared_projection_models(cfg)
        return self.__build_separate_projection_models(cfg)

    def __build_separate_projection_models(self, cfg: "ModelConfig"):
        query_model = self.model_type.value(cfg)
        key_model = self.model_type.value(cfg)
        value_model = self.model_type.value(cfg)
        output_model = self.model_type.value(cfg)
        self.register_parameter("qkv_model", None)
        return query_model, key_model, value_model, output_model

    def __build_shared_projection_models(self, cfg: "ModelConfig"):
        self.register_parameter("query_model", None)
        self.register_parameter("key_model", None)
        self.register_parameter("value_model", None)
        output_model = self.model_type.value(cfg)
        qkv_model = self.model_type.value(cfg)
        return qkv_model, output_model

    def __are_qkv_dimensions_equal(self) -> bool:
        are_keys_querys_same = self.key_dim == self.embedding_dim
        are_values_querys_same = self.value_dim == self.embedding_dim
        return are_keys_querys_same and are_values_querys_same

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
        query, key, value = self.utils.maybe_transpose_qkv(query, key, value)
        self.validator.multi_head_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )
        query, key, value, key_padding_mask, attention_mask = (
            self.utils.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )
        key_padding_mask, attention_mask = (
            self.masks.validate_padding_and_attention_masks(
                attention_mask,
                key_padding_mask,
                need_weights,
            )
        )

        query, key, value = self.projector.compute_qkv_projections(query, key, value)
        (
            key,
            value,
            key_padding_mask,
            attention_mask,
        ) = self.utils.add_bias_vectors_to_kv(
            key,
            value,
            key_padding_mask,
            attention_mask,
        )
        query, key, value = self.utils.prepare_qkv_projection_for_attention(
            query, key, value, static_key, static_values
        )
        key, value, attention_mask, key_padding_mask = self.utils.add_zero_attention(
            key, value, attention_mask, key_padding_mask
        )
        updated_source_sequence_length = key.size(1)
        merged_mask = self.masks.merge_masks(attention_mask, key_padding_mask)
        attention_output, attention_weights = self.attention_computation.compute_(
            query, key, value, merged_mask
        )
        return attention_output, attention_weights
