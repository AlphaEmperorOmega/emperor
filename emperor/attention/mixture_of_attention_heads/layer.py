from emperor.attention.core.layers import MultiHeadAttentionAbstract
from emperor.attention.mixture_of_attention_heads.validator import (
    MixtureOfAttentionHeadsValidator,
)
from emperor.attention.mixture_of_attention_heads.projector import (
    MixtureOfAttentionHeadsProjector,
)
from emperor.attention.mixture_of_attention_heads.processor import (
    MixtureOfAttentionHeadsProcessor,
)
from emperor.attention.mixture_of_attention_heads.reshaper import (
    MixtureOfAttentionHeadsReshaper,
)
from emperor.attention.mixture_of_attention_heads.mask import (
    MixtureOfAttentionHeadsMask,
)


class MixtureOfAttentionHeads(MultiHeadAttentionAbstract):
    VALIDATOR = MixtureOfAttentionHeadsValidator

    def _build_attention_components(self) -> None:
        self.projector = MixtureOfAttentionHeadsProjector(self.cfg)
        self.reshaper = MixtureOfAttentionHeadsReshaper(self.cfg)
        self.processor = MixtureOfAttentionHeadsProcessor(
            self.cfg, self.projector, self.reshaper
        )
        self.masks = MixtureOfAttentionHeadsMask(self.cfg)
