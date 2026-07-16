"""Private self-attention layer implementation."""

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._ops.masking import Mask
from emperor.attention._ops.reshaping import AttentionReshaper
from emperor.attention._variants.self_attention.processing import (
    SelfAttentionProcessor,
)
from emperor.attention._variants.self_attention.projection import (
    SelfAttentionProjector,
)
from emperor.attention._variants.self_attention.validation import (
    SelfAttentionValidator,
)


class SelfAttention(MultiHeadAttentionAbstract):
    VALIDATOR = SelfAttentionValidator

    def _build_attention_components(self) -> None:
        self.projector = SelfAttentionProjector(self.cfg)
        self.reshaper = AttentionReshaper(self.cfg)
        self.processor = SelfAttentionProcessor(self.cfg, self.projector, self.reshaper)
        self.masks = Mask(self.cfg)
