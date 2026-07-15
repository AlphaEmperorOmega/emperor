from emperor.attention.core.handlers.mask import Mask
from emperor.attention.core.handlers.reshaper import AttentionReshaper
from emperor.attention.core.layers import MultiHeadAttentionAbstract
from emperor.attention.core.variants.independent_attention.processor import (
    IndependentProcessor,
)
from emperor.attention.core.variants.independent_attention.projector import (
    IndependentProjector,
)
from emperor.attention.core.variants.independent_attention.validator import (
    IndependentAttentionValidator,
)


class IndependentAttention(MultiHeadAttentionAbstract):
    VALIDATOR = IndependentAttentionValidator

    def _build_attention_components(self) -> None:
        self.projector = IndependentProjector(self.cfg)
        self.reshaper = AttentionReshaper(self.cfg)
        self.processor = IndependentProcessor(self.cfg, self.projector, self.reshaper)
        self.masks = Mask(self.cfg)
