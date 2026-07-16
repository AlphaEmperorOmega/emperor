from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerConfig, ResidualConnectionOptions

if TYPE_CHECKING:
    from emperor.attention import (
        IndependentAttentionConfig,
        MixtureOfAttentionHeadsConfig,
        SelfAttentionConfig,
    )
    from emperor.layers import LayerNormPositionOptions
    from emperor.transformer._feed_forward import FeedForwardConfig

    DecoderSelfAttentionConfig = SelfAttentionConfig | MixtureOfAttentionHeadsConfig
    DecoderCrossAttentionConfig = (
        IndependentAttentionConfig | MixtureOfAttentionHeadsConfig
    )


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
    attention_config: "SelfAttentionConfig | MixtureOfAttentionHeadsConfig | None" = (
        optional_field("Supported attention configuration for the encoder sub-block.")
    )
    feed_forward_config: "FeedForwardConfig | None" = optional_field(
        "Feed-forward configuration for the encoder sub-block."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer._layers import TransformerEncoderLayer

        return TransformerEncoderLayer


@dataclass
class TransformerEncoderBlockLayerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.transformer._layers import TransformerEncoderBlockLayer

        return TransformerEncoderBlockLayer


@dataclass
class TransformerDecoderBlockLayerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.transformer._layers import TransformerDecoderBlockLayer

        return TransformerDecoderBlockLayer


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
    self_attention_config: "DecoderSelfAttentionConfig | None" = optional_field(
        "Self-attention configuration for the decoder sub-block."
    )
    cross_attention_config: "DecoderCrossAttentionConfig | None" = optional_field(
        "Cross-attention configuration querying encoder outputs."
    )
    feed_forward_config: "FeedForwardConfig | None" = optional_field(
        "Feed-forward configuration for the decoder sub-block."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer._layers import TransformerDecoderLayer

        return TransformerDecoderLayer


@dataclass
class TransformerEncoderStackConfig(ConfigBase):
    num_layers: int | None = optional_field("Number of encoder layers in the stack.")
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
        from emperor.transformer._stacks import TransformerEncoderStack

        return TransformerEncoderStack


@dataclass
class TransformerDecoderStackConfig(ConfigBase):
    num_layers: int | None = optional_field("Number of decoder layers in the stack.")
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
        from emperor.transformer._stacks import TransformerDecoderStack

        return TransformerDecoderStack


@dataclass
class TransformerConfig(ConfigBase):
    encoder_stack_config: "TransformerEncoderStackConfig | None" = optional_field(
        "Encoder stack configuration."
    )
    decoder_stack_config: "TransformerDecoderStackConfig | None" = optional_field(
        "Decoder stack configuration."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer._model import Transformer

        return Transformer
