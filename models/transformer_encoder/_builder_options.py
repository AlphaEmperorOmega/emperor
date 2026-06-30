from dataclasses import dataclass

from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig


@dataclass(frozen=True)
class TransformerEncoderOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions
    causal_attention_mask_flag: bool = False


@dataclass(frozen=True)
class TransformerPositionalEmbeddingOptions:
    option: type[AbsolutePositionalEmbeddingConfig]
    padding_idx: int | None
    auto_expand_flag: bool


@dataclass(frozen=True)
class TransformerAttentionOptions:
    num_heads: int
    num_layers: int
    bias_flag: bool
    add_key_value_bias_flag: bool


@dataclass(frozen=True)
class TransformerFeedForwardOptions:
    num_layers: int
    bias_flag: bool


@dataclass(frozen=True)
class VitPatchOptions:
    patch_size: int
    input_channels: int
    image_height: int
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True)
class VitOutputOptions:
    num_layers: int
    bias_flag: bool
