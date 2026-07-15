from dataclasses import dataclass

from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.base.config import optional_field
from emperor.base.options import BaseOptions


class SelfAttentionProjectionStrategy(BaseOptions):
    FUSED = 0
    SEPARATE = 1


@dataclass
class SelfAttentionConfig(MultiHeadAttentionConfig):
    projection_strategy: SelfAttentionProjectionStrategy | None = optional_field(
        "Whether Q/K/V use one fused projection model or three separate models."
    )

    def _registry_owner(self) -> type:
        from emperor.attention.core.variants.self_attention.layer import SelfAttention

        return SelfAttention
