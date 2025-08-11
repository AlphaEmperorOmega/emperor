import torch
import copy
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
from Emperor.base.utils import DataClassBase, Module, ParameterBank, device

from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import (
    LayerTypes,
)

from typing import TYPE_CHECKING

from Emperor.layers.utils.linears import LinearLayer

if TYPE_CHECKING:
    from torch.types import _dtype as DType
    from Emperor.config import ModelConfig


@dataclass
class MultiHeadAttentionConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={
            "help": "Type of model used to generate parameters query, key, and value projections"
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Number of samples (batch size) processed in parallel during training/inference."
        },
    )
    num_heads: int | None = field(
        default=None,
        metadata={"help": "Number of attention heads to use for multi-head attention."},
    )
    embedding_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the input embedding vector (input feature size)."
        },
    )
    key_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the key vectors (defaults to embedding_dim if 0)."
        },
    )
    value_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the value vectors (defaults to embedding_dim if 0)."
        },
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={
            "help": "Length of the target sequence for decoding or output (number of target tokens)."
        },
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={
            "help": "Length of the source/input sequence (number of input tokens)."
        },
    )
    target_dtype: "DType | None" = field(
        default=None,
        metadata={
            "help": "Data type (dtype) for the attention outputs (e.g. torch.float32)."
        },
    )
    use_separate_projection_weight: bool | None = field(
        default=None,
        metadata={
            "help": "If True, use separate projection weights for Q, K, V. If False, use shared projection weights (single QKV matrix)."
        },
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={
            "help": "Dropout probability applied to attention weights (prevents overfitting)."
        },
    )
    key_value_bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, add learnable bias vectors to key and value projections."
        },
    )
    zero_attention_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, add zero vectors to attention keys/values to allow explicit non-attending positions."
        },
    )
    batch_first_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, input/output tensors are in (batch, seq, feature) format. If False, uses (seq, batch, feature)."
        },
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, use a causal mask to prevent attention to future positions (for decoding/generation)."
        },
    )
    add_key_value_bias_flag: bool | None = field(
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

        self.main_cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.model_type = self.cfg.model_type
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
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
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.__resolve_kv_dimensions()
        self._valudate_fields(self.cfg, MultiHeadAttentionConfig)

        m = self.__build_projection_models()
        if len(m) == 4:
            self.query_model, self.key_model, self.value_model, self.output_model = m
        else:
            self.qkv_model, self.output_model = m
        self.key_bias_vector, self.value_bias_vector = self.__build_kv_bias_vectors()

        self.__initialize_attention_components()
        self.head_dim = self.__resolve_head_dim()

    def __resolve_head_dim(self):
        head_dim = self.embedding_dim // self.num_heads
        self.validator.assert_correct_head_dim(head_dim)
        return head_dim

    def __resolve_kv_dimensions(self):
        self.key_dim = self.embedding_dim if self.key_dim == 0 else self.key_dim
        self.value_dim = self.embedding_dim if self.value_dim == 0 else self.value_dim

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
        self.utils = AttentionUtils(
            self.cfg,
            self.validator,
            self.key_bias_vector,
            self.value_bias_vector,
        )

    def __build_projection_models(self) -> tuple:
        if (
            not self.use_separate_projection_weight
            and self.__are_qkv_dimensions_equal()
        ):
            return self.__build_shared_projection_models()
        return self.__build_separate_projection_models()

    def __build_kv_bias_vectors(self):
        if self.add_key_value_bias_flag:
            bias_k = self._init_parameter_bank((1, 1, self.embedding_dim))
            bias_v = self._init_parameter_bank((1, 1, self.embedding_dim))
            return bias_k, bias_v
        return None, None

    def __build_separate_projection_models(self) -> tuple:
        query_model = self.__create_model(self.embedding_dim, self.key_dim)
        key_model = self.__create_model(self.embedding_dim, self.key_dim)
        value_model = self.__create_model(self.embedding_dim, self.value_dim)
        output_model = self.__create_model(self.value_dim, self.embedding_dim)
        self.register_parameter("qkv_model", None)
        return query_model, key_model, value_model, output_model

    def __build_shared_projection_models(self) -> tuple:
        self.register_parameter("query_model", None)
        self.register_parameter("key_model", None)
        self.register_parameter("value_model", None)
        output_model = self.__create_model(self.embedding_dim, self.embedding_dim)
        qkv_model = self.__create_model(self.embedding_dim, self.embedding_dim * 3)
        return qkv_model, output_model

    def __create_model(self, input_dim: int, output_dim: int):
        config = self.__resolve_model_type_overrides(
            self.main_cfg, input_dim, output_dim
        )
        output_model = self.model_type.value(config)
        return LayerBlock(model=output_model)

    def __resolve_model_type_overrides(
        self,
        cfg: "ModelConfig",
        input_dim: int,
        output_dim: int,
    ):
        c = copy.deepcopy(cfg)
        if issubclass(self.model_type.value, LinearLayer):
            c.linear_layer_model_config.input_dim = input_dim
            c.linear_layer_model_config.output_dim = output_dim
            return c
        c.mixture_model_config.input_dim = input_dim
        c.mixture_model_config.output_dim = output_dim
        return c

    def __are_qkv_dimensions_equal(self) -> bool:
        are_qk_dims_same = self.embedding_dim == self.key_dim
        are_qv_dims_same = self.embedding_dim == self.value_dim
        return are_qk_dims_same and are_qv_dims_same

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
        nn.MultiheadAttention
        query, key, value = self.projector.compute_qkv_projections(query, key, value)
        (
            key,
            value,
            key_padding_mask,
            attention_mask,
        ) = self.utils.add_learnable_bias_vectors(
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
