from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerConfig

if TYPE_CHECKING:
    from emperor.attention import (
        IndependentAttentionConfig,
        MixerAttentionConfig,
        MixtureOfAttentionHeadsConfig,
        SelfAttentionConfig,
    )
    from emperor.layers import (
        LayerNormPositionOptions,
        LayerStackConfig,
        RecurrentLayerConfig,
        ResidualConfig,
    )
    from emperor.transformer._feed_forward import FeedForwardConfig

    DecoderSelfAttentionConfig = (
        SelfAttentionConfig | MixtureOfAttentionHeadsConfig | MixerAttentionConfig
    )
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
    residual_config: "ResidualConfig | None" = optional_field(
        "Optional residual connection config applied to every encoder sub-block join."
    )
    attention_config: "SelfAttentionConfig | MixtureOfAttentionHeadsConfig | MixerAttentionConfig | None" = optional_field(  # noqa: E501
        "Supported self-processing configuration for the encoder attention sub-block."
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
    residual_config: "ResidualConfig | None" = optional_field(
        "Optional residual connection config applied to every decoder sub-block join."
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
class TransformerDecoderBlockLayerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.transformer._layers import TransformerDecoderBlockLayer

        return TransformerDecoderBlockLayer


@dataclass
class TransformerConfig(ConfigBase):
    encoder_stack_config: "LayerStackConfig | RecurrentLayerConfig | None" = (
        optional_field("Generic encoder block-stack configuration.")
    )
    decoder_stack_config: "LayerStackConfig | RecurrentLayerConfig | None" = (
        optional_field("Generic decoder block-stack configuration.")
    )

    def _registry_owner(self) -> type:
        from emperor.transformer._model import Transformer

        return Transformer
