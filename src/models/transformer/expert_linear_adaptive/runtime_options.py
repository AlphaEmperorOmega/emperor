from __future__ import annotations

from dataclasses import dataclass

from emperor.embedding.absolute.core.config import (
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
class ExpertOptions:
    num_experts: int = 4
    top_k: int = 2
    normalize_probabilities_flag: bool = True
    switch_loss_weight: float = 0.0
    capacity_factor: float = 0.0


@dataclass(frozen=True)
class AdaptiveParameterOptions:
    weight_option: type | None = None
    bias_option: type | None = None
    diagonal_option: type | None = None
    row_mask_option: type | None = None


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
    attention_expert_options: ExpertOptions = ExpertOptions()
    feed_forward_expert_options: ExpertOptions = ExpertOptions()
    attention_projection_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    attention_expert_adaptive_options: AdaptiveParameterOptions = (
        AdaptiveParameterOptions()
    )
    router_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()
    feed_forward_adaptive_options: AdaptiveParameterOptions = AdaptiveParameterOptions()


__all__ = [
    "AdaptiveParameterOptions",
    "ControllerStackOptions",
    "DynamicMemoryOptions",
    "ExpertOptions",
    "LayerControllerOptions",
    "RecurrentControllerOptions",
    "RuntimeOptions",
    "SubmoduleStackOptions",
    "SubmoduleStackSource",
    "TransformerAttentionOptions",
    "TransformerFeedForwardOptions",
    "TransformerStackOptions",
]
