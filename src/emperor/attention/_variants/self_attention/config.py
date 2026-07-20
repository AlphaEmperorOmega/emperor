"""Private self-attention configuration implementation."""

from dataclasses import dataclass

from emperor.attention._config import MultiHeadAttentionConfig
from emperor.config import BaseOptions, optional_field


class SelfAttentionProjectionStrategy(BaseOptions):
    FUSED = 0
    SEPARATE = 1
    FUSED_KEY_VALUE = 2


@dataclass
class SelfAttentionConfig(MultiHeadAttentionConfig):
    projection_strategy: SelfAttentionProjectionStrategy | None = optional_field(
        "Whether Q/K/V use one fused model, a Q model plus a fused K/V model, "
        "or three separate models."
    )

    def _registry_owner(self) -> type:
        from emperor.attention._variants.self_attention.layer import SelfAttention

        return SelfAttention
