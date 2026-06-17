from dataclasses import dataclass
from emperor.attention.core.config import MultiHeadAttentionConfig


@dataclass
class IndependentAttentionConfig(MultiHeadAttentionConfig):
    def _registry_owner(self) -> type:
        from emperor.attention.core.variants.independent_attention.layer import (
            IndependentAttention,
        )

        return IndependentAttention
