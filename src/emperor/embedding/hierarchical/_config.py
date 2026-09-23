from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class HierarchicalByteEmbeddingConfig(ConfigBase):
    byte_embedding_dim: int | None = optional_field("Width of each byte feature.")
    output_dim: int | None = optional_field("Width of each output token embedding.")
    max_token_bytes: int | None = optional_field(
        "Maximum UTF-8 bytes per token, excluding the leading pooling symbol."
    )
    byte_position_config: ConfigBase | None = optional_field(
        "Text positional embedding configuration for positions within each token."
    )
    encoder_config: ConfigBase | None = optional_field(
        "Encoder-only Transformer configuration with batch-first bidirectional attention."
    )
    projection_config: ConfigBase | None = optional_field(
        "LayerState model configuration mapping pooled byte features to output_dim."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.hierarchical._component import HierarchicalByteEmbedding

        return HierarchicalByteEmbedding
