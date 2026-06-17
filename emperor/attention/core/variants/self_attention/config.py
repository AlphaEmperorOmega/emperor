from dataclasses import dataclass
from emperor.attention.core.config import MultiHeadAttentionConfig


@dataclass
class SelfAttentionConfig(MultiHeadAttentionConfig):
    def _registry_owner(self) -> type:
        from emperor.attention.core.variants.self_attention.layer import SelfAttention

        return SelfAttention
