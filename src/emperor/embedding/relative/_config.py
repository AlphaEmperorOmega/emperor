from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field


@dataclass
class RelativePositionalEmbeddingConfig(ConfigBase):
    num_heads: int | None = optional_field("Number of attention heads.")
    embedding_dim: int | None = optional_field("Attention embedding dimension.")
    max_positions: int | None = optional_field(
        "Maximum relative distance represented directly."
    )


@dataclass
class DynamicPositionalBiasConfig(RelativePositionalEmbeddingConfig):
    def _registry_owner(self) -> type:
        from emperor.embedding.relative._variants.bias import DynamicPositionalBias

        return DynamicPositionalBias
