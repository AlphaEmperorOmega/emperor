"""Private independent-attention layer implementation."""

from emperor.attention._base import MultiHeadAttentionAbstract
from emperor.attention._ops.masking import Mask
from emperor.attention._ops.reshaping import AttentionReshaper
from emperor.attention._variants.independent.processing import IndependentProcessor
from emperor.attention._variants.independent.projection import IndependentProjector
from emperor.attention._variants.independent.validation import (
    IndependentAttentionValidator,
)


class IndependentAttention(MultiHeadAttentionAbstract):
    VALIDATOR = IndependentAttentionValidator

    def _build_attention_components(self) -> None:
        self.projector = IndependentProjector(self.cfg)
        self.reshaper = AttentionReshaper(self.cfg)
        self.processor = IndependentProcessor(self.cfg, self.projector, self.reshaper)
        self.masks = Mask(self.cfg)
