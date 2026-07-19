from __future__ import annotations

from dataclasses import dataclass

from emperor.embedding.absolute import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.transformer import (
    ControllerStackOptions,
    DynamicMemoryOptions,
    LayerControllerOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    SubmoduleStackSource,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
)


@dataclass(frozen=True)
class RuntimeOptions:
    batch_size: int = 64
    learning_rate: float = 1.0
    vocab_size: int = 8192
    model_dim: int = 128
    source_sequence_length: int = 64
    target_sequence_length: int = 64
    dropout_probability: float = 0.1
    positional_embedding_option: type = TextSinusoidalPositionalEmbeddingConfig
    encoder_options: TransformerStackOptions = TransformerStackOptions()
    decoder_options: TransformerStackOptions = TransformerStackOptions()
    encoder_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_self_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    decoder_cross_attention_options: TransformerAttentionOptions = (
        TransformerAttentionOptions()
    )
    encoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )
    decoder_feed_forward_options: TransformerFeedForwardOptions = (
        TransformerFeedForwardOptions()
    )


__all__ = [
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "RuntimeOptions",
    "SubmoduleStackOptions",
    "SubmoduleStackSource",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerStackOptions",
]
