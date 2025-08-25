from torch import Tensor
from dataclasses import dataclass, field
from Emperor.attention.utils.utils import (
    AttentionKeyValueBias,
    AttentionMask,
    AttentionProcessor,
    AttentionProjector,
    AttentionUtils,
    AttentionValidator,
)
from Emperor.base.utils import DataClassBase, Module
from Emperor.layers.utils.enums import LayerTypes

from typing import TYPE_CHECKING

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
    query_key_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the query and key projected vectors (defaults to embedding_dim if 0)."
        },
    )
    value_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of the value projected vectors (defaults to embedding_dim if 0)."
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
    use_separate_projection_weight_flag: bool | None = field(
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
    average_attention_weights_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    return_attention_weights_flag: bool | None = field(
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
        self.num_heads = self.cfg.num_heads
        self.batch_size = self.cfg.batch_size
        self.model_type = self.cfg.model_type
        self.target_dtype = self.cfg.target_dtype
        self.embedding_dim = self.cfg.embedding_dim
        self.batch_first_flag = self.cfg.batch_first_flag
        self.dropout_probability = self.cfg.dropout_probability
        self.key_value_bias_flag = self.cfg.key_value_bias_flag
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.value_projection_dim = self.cfg.value_projection_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.use_separate_projection_weight_flag = (
            self.cfg.use_separate_projection_weight_flag
        )
        self.average_attention_weights_flag = self.cfg.average_attention_weights_flag
        self._validate_fields(self.cfg, MultiHeadAttentionConfig)
        self.__create_attention_utilities()
        self.head_dim = self.__resolve_head_dim()

    def __resolve_head_dim(self):
        head_dim = self.embedding_dim // self.num_heads
        self.validator.assert_correct_head_dim(head_dim)
        return head_dim

    def __create_attention_utilities(self):
        self.validator = AttentionValidator(self.cfg)
        self.masks = AttentionMask(self.cfg, self.validator)
        self.projector = AttentionProjector(self.cfg, self.main_cfg, self.validator)
        self.processor = AttentionProcessor(self.cfg, self.validator, self.projector)
        self.learnable_bias_handler = AttentionKeyValueBias(self.cfg)
        self.utils = AttentionUtils(
            self.cfg,
            self.validator,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_key: Tensor | None = None,
        static_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
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
                key_padding_mask,
                attention_mask,
            )
        )
        query, key, value = self.projector.compute_qkv_projections(query, key, value)
        (
            key,
            value,
            key_padding_mask,
            attention_mask,
        ) = self.learnable_bias_handler.add_kv_learnable_bias_vectors(
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
        merged_mask = self.utils.merge_masks(key, key_padding_mask, attention_mask)
        attention_output, attention_weights = self.processor.compute_attention(
            query, key, value, merged_mask
        )
        if self.batch_first_flag and self.validator.is_input_batched():
            attention_output = attention_output.transpose(1, 0)
        return attention_output, attention_weights
