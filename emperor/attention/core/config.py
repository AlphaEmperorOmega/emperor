from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
    from emperor.base.layer import LayerStackConfig
    from emperor.embedding.relative.config import RelativePositionalEmbeddingConfig


@dataclass
class MultiHeadAttentionConfig(ConfigBase):
    batch_size: int | None = optional_field(
        "Number of samples processed in parallel during training/inference."
    )
    num_heads: int | None = optional_field(
        "Number of attention heads to use for multi-head attention."
    )
    embedding_dim: int | None = optional_field(
        "Dimension of the input embedding vector (input feature size)."
    )
    query_key_projection_dim: int | None = optional_field(
        "Dimension of the query and key projected vectors (defaults to embedding_dim if 0)."
    )
    value_projection_dim: int | None = optional_field(
        "Dimension of the value projected vectors (defaults to embedding_dim if 0)."
    )
    target_sequence_length: int | None = optional_field(
        "Length of the target sequence for decoding or output (number of target tokens)."
    )
    source_sequence_length: int | None = optional_field(
        "Length of the source/input sequence (number of input tokens)."
    )
    target_dtype: "DType | None" = optional_field(
        "Data type for the attention outputs (e.g. torch.float32)."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability applied to attention weights (prevents overfitting)."
    )
    zero_attention_flag: bool | None = optional_field(
        "If True, add zero vectors to attention keys/values to allow explicit non-attending positions."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "If True, use a causal mask to prevent attention to future positions."
    )
    add_key_value_bias_flag: bool | None = optional_field(
        "If True, add learnable bias vectors to the key and value projections."
    )
    average_attention_weights_flag: bool | None = optional_field(
        "If True, average the returned attention weights across heads."
    )
    return_attention_weights_flag: bool | None = optional_field(
        "If True, return the attention weights alongside the attention output."
    )
    projection_model_config: "LayerStackConfig | None" = optional_field(
        "Layer-stack configuration used to build the query/key/value/output projections."
    )
    relative_positional_embedding_config: "RelativePositionalEmbeddingConfig | None" = (
        optional_field("Configuration for the relative positional embedding module.")
    )
