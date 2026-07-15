from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.base.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from torch.types import _dtype as DType

    from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
    from emperor.embedding.relative.core.config import RelativePositionalEmbeddingConfig


@dataclass
class MultiHeadAttentionConfig(ConfigBase):
    batch_size: int | None = optional_field(
        "Maximum runtime batch size accepted by this attention layer."
    )
    num_heads: int | None = optional_field(
        "Number of attention heads to use for multi-head attention."
    )
    embedding_dim: int | None = optional_field(
        "Dimension of the input embedding vector (input feature size)."
    )
    query_key_projection_dim: int | None = optional_field(
        "Dimension of the query and key projected vectors "
        "(defaults to embedding_dim if 0)."
    )
    value_projection_dim: int | None = optional_field(
        "Dimension of the value projected vectors (defaults to embedding_dim if 0)."
    )
    target_sequence_length: int | None = optional_field(
        "Maximum runtime target/query sequence length."
    )
    source_sequence_length: int | None = optional_field(
        "Maximum runtime selected key/value source sequence length."
    )
    target_dtype: "DType | None" = optional_field(
        "Data type for the attention outputs (e.g. torch.float32)."
    )
    dropout_probability: float | None = optional_field(
        "Dropout probability applied to attention weights (prevents overfitting)."
    )
    zero_attention_flag: bool | None = optional_field(
        "If True, add zero vectors to attention keys/values to allow explicit "
        "non-attending positions."
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
    batch_first_flag: bool | None = optional_field(
        "Explicit input layout for batched tensors. True selects [batch, sequence, "
        "embedding], False selects [sequence, batch, embedding], and None preserves "
        "legacy configured-batch-size detection."
    )
    projection_model_config: "LayerStackConfig | RecurrentLayerConfig | None" = (
        optional_field(
            "Projection-model configuration used to build the query/key/value/output "
            "projections."
        )
    )
    relative_positional_embedding_config: "RelativePositionalEmbeddingConfig | None" = (
        optional_field("Configuration for the relative positional embedding module.")
    )
