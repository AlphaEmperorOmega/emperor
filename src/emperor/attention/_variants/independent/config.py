"""Private independent-attention configuration implementation."""

from dataclasses import dataclass

from emperor.attention._config import MultiHeadAttentionConfig


@dataclass
class IndependentAttentionConfig(MultiHeadAttentionConfig):
    def _registry_owner(self) -> type:
        from emperor.attention._variants.independent.layer import IndependentAttention

        return IndependentAttention
