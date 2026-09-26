from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class HierarchicalByteDecoderConfig(ConfigBase):
    """Explicit child composition for the HAT v2 adaptation, not a paper preset.

    Required fields must be supplied before build. See
    docs/hierarchical-language-model.md for shapes, loss and generation contracts.
    """

    conditioning_dim: int | None = optional_field(
        "Width of predictive context features."
    )
    byte_embedding_dim: int | None = optional_field("Width of decoder byte features.")
    max_token_bytes: int | None = optional_field("Maximum bytes in a token prefix.")
    byte_position_config: ConfigBase | None = optional_field(
        "Learned or sinusoidal positions within the context-prefixed byte sequence."
    )
    decoder_config: ConfigBase | None = optional_field(
        "Decoder-only Transformer with batch-first causal self-attention."
    )
    conditioning_projection_config: ConfigBase | None = optional_field(
        "LayerState model mapping conditioning_dim to byte_embedding_dim."
    )
    output_projection_config: ConfigBase | None = optional_field(
        "LayerState model mapping byte_embedding_dim to 258 byte/control logits."
    )

    def _registry_owner(self) -> type:
        from ._component import HierarchicalByteDecoder

        return HierarchicalByteDecoder


@dataclass
class HierarchicalLanguageModelConfig(ConfigBase):
    """Supplied encoder, causal backbone and byte head; no word vocabulary.

    Adds a learned beginning-of-document context. Architecture choices remain
    owned by the supplied configurations, not inferred from the HAT paper.
    """

    sequence_length: int | None = optional_field("Maximum rolling backbone context.")
    embedding_config: ConfigBase | None = optional_field(
        "Hierarchical byte encoder configuration."
    )
    backbone_config: ConfigBase | None = optional_field(
        "Causal Transformer decoder stack configuration."
    )
    decoding_config: HierarchicalByteDecoderConfig | None = optional_field(
        "Conditional byte decoder configuration."
    )
    position_config: ConfigBase | None = optional_field(
        "Positions on the backbone token sequence."
    )
    embedding_normalization_config: ConfigBase | None = optional_field(
        "Optional LayerConfig for input normalization."
    )
    output_normalization_config: ConfigBase | None = optional_field(
        "Optional LayerConfig for backbone output normalization."
    )
    dropout_probability: float | None = optional_field(
        "Dropout on backbone input embeddings."
    )

    def _registry_owner(self) -> type:
        from ._language_model import HierarchicalLanguageModel

        return HierarchicalLanguageModel
