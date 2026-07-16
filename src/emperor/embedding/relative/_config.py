from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class RelativePositionalEmbeddingConfig(ConfigBase):
    text_processing_flag: bool | None = optional_field(
        "Whether the relative embedding is used for text processing."
    )
    num_heads: int | None = optional_field("Number of attention heads.")
    num_embeddings: int | None = optional_field(
        "Number of relative positional embeddings."
    )
    embedding_dim: int | None = optional_field("Attention embedding dimension.")
    init_size: int | None = optional_field("Initial embedding table size.")
    padding_idx: int | None = optional_field("Optional padding index.")
    auto_expand_flag: bool | None = optional_field(
        "Whether the embedding table can expand at runtime."
    )
    max_positions: int | None = optional_field(
        "Maximum relative distance represented directly."
    )


@dataclass
class DynamicPositionalBiasConfig(RelativePositionalEmbeddingConfig):
    def _registry_owner(self) -> type:
        from emperor.embedding.relative._bias import DynamicPositionalBias

        return DynamicPositionalBias
