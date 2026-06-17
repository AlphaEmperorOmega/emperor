from dataclasses import dataclass
from emperor.base.layer import LayerConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.utils import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.options import LayerNormPositionOptions
    from emperor.attention.core.variants.self_attention.config import SelfAttentionConfig
    from emperor.attention.core.variants.independent_attention.config import (
        IndependentAttentionConfig,
    )
    from emperor.transformer.feed_forward.core.config import FeedForwardConfig


@dataclass
class TransformerEncoderLayerConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Token embedding dimension shared by attention and feed-forward."
    )
    layer_norm_position: "LayerNormPositionOptions | None" = optional_field(
        "Where layer normalization is applied within each sub-block."
    )
    dropout_probability: float | None = optional_field(
        "Dropout applied after each sub-block. Use 0.0 to disable."
    )
    residual_connection_option: ResidualConnectionOptions | None = optional_field(
        "Residual connection behavior applied to every encoder sub-block join."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "Force a causal attention mask in the stack-level mask generation."
    )
    attention_config: "SelfAttentionConfig | None" = optional_field(
        "Self-attention configuration for the encoder sub-block."
    )
    feed_forward_config: "FeedForwardConfig | None" = optional_field(
        "Feed-forward configuration for the encoder sub-block."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.core.layers import TransformerEncoderLayer

        return TransformerEncoderLayer


@dataclass
class TransformerEncoderBlockLayerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.transformer.core.layers import TransformerEncoderBlockLayer

        return TransformerEncoderBlockLayer


@dataclass
class TransformerDecoderLayerConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Token embedding dimension shared by attention and feed-forward."
    )
    layer_norm_position: "LayerNormPositionOptions | None" = optional_field(
        "Where layer normalization is applied within each sub-block."
    )
    dropout_probability: float | None = optional_field(
        "Dropout applied after each sub-block. Use 0.0 to disable."
    )
    residual_connection_option: ResidualConnectionOptions | None = optional_field(
        "Residual connection behavior applied to every decoder sub-block join."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "Force a causal attention mask in the stack-level mask generation."
    )
    self_attention_config: "SelfAttentionConfig | None" = optional_field(
        "Self-attention configuration for the decoder sub-block."
    )
    cross_attention_config: "IndependentAttentionConfig | None" = optional_field(
        "Cross-attention configuration querying encoder outputs."
    )
    feed_forward_config: "FeedForwardConfig | None" = optional_field(
        "Feed-forward configuration for the decoder sub-block."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.core.layers import TransformerDecoderLayer

        return TransformerDecoderLayer


@dataclass
class TransformerEncoderStackConfig(ConfigBase):
    num_layers: int | None = optional_field(
        "Number of encoder layers in the stack."
    )
    embedding_dim: int | None = optional_field(
        "Token embedding dimension. Drives final layer-norm sizing."
    )
    source_sequence_length: int | None = optional_field(
        "Source sequence length used by causal mask generation."
    )
    target_sequence_length: int | None = optional_field(
        "Target sequence length, asserted equal to source for encoder."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "Force causal masking when generating attention masks."
    )
    layer_config: "TransformerEncoderLayerConfig | None" = optional_field(
        "Encoder layer config used to build each layer in the stack."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.core.stack import TransformerEncoderStack

        return TransformerEncoderStack


@dataclass
class TransformerDecoderStackConfig(ConfigBase):
    num_layers: int | None = optional_field(
        "Number of decoder layers in the stack."
    )
    embedding_dim: int | None = optional_field(
        "Token embedding dimension. Drives final layer-norm sizing."
    )
    source_sequence_length: int | None = optional_field(
        "Source sequence length used by causal mask generation."
    )
    target_sequence_length: int | None = optional_field(
        "Target sequence length used by causal mask generation."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "Force causal masking when generating attention masks."
    )
    layer_config: "TransformerDecoderLayerConfig | None" = optional_field(
        "Decoder layer config used to build each layer in the stack."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.core.stack import TransformerDecoderStack

        return TransformerDecoderStack
