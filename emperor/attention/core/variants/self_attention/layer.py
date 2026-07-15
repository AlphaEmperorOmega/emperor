from emperor.attention.core.handlers.mask import Mask
from emperor.attention.core.handlers.reshaper import AttentionReshaper
from emperor.attention.core.layers import MultiHeadAttentionAbstract
from emperor.attention.core.variants.self_attention.processor import (
    SelfAttentionProcessor,
)
from emperor.attention.core.variants.self_attention.projector import (
    SelfAttentionProjector,
)
from emperor.attention.core.variants.self_attention.validator import (
    SelfAttentionValidator,
)


class SelfAttention(MultiHeadAttentionAbstract):
    VALIDATOR = SelfAttentionValidator

    def _build_attention_components(self) -> None:
        self.projector = SelfAttentionProjector(self.cfg)
        self.reshaper = AttentionReshaper(self.cfg)
        self.processor = SelfAttentionProcessor(self.cfg, self.projector, self.reshaper)
        self.masks = Mask(self.cfg)
