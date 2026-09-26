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
