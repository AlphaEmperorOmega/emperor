from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
    from emperor.embedding.relative import DynamicPositionalBiasConfig
    from emperor.experts import MixtureOfExpertsConfig


@dataclass
class CausalPrefixKernelConfig(ConfigBase):
    hidden_dim: int | None = optional_field("Input and output feature dimension.")
    kernel_dim: int = field(
        default=32,
        metadata={"help": "Query, key, and value feature dimension."},
    )
    relative_position_config: "DynamicPositionalBiasConfig | None" = optional_field(
        "One-head relative positional bias configuration."
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.contextual._kernel import CausalPrefixKernel

        return CausalPrefixKernel


@dataclass
class ByteContextualEmbeddingConfig(ConfigBase):
    max_token_bytes: int = field(
        default=4,
        metadata={"help": "Maximum UTF-8 bytes retained for each token."},
    )
    hidden_dim: int | None = optional_field("Contextual embedding dimension.")
    byte_moe_config: "MixtureOfExpertsConfig | None" = optional_field(
        "Independent mixture-of-experts configuration for byte features."
    )
    positional_embedding_config: "TextLearnedPositionalEmbeddingConfig | None" = (
        optional_field("Learned text positional embedding configuration.")
    )
    prefix_kernel_config: CausalPrefixKernelConfig | None = optional_field(
        "Inclusive-causal prefix kernel configuration."
    )
    context_moe_config: "MixtureOfExpertsConfig | None" = optional_field(
        "Independent mixture-of-experts configuration for contextual features."
    )
    residual_scale_initial_value: float = field(
        default=1e-3,
        metadata={"help": "Initial scalar multiplier for the contextual correction."},
    )

    def _registry_owner(self) -> type:
        from emperor.embedding.contextual._component import ByteContextualEmbedding

        return ByteContextualEmbedding
